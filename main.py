import argparse
import collections
import json
import math
import os
import shutil
import sys

import jsonlines
import numpy as np

MINOVERLAP = 0.5 # default value (defined in the PASCAL VOC2012 challenge)

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output-dir', required=False, default="/tmp", help="dir to output files")
parser.add_argument('-gt', '--ground-truth', required=True, help="dir to ground truth annotations")
parser.add_argument('-r', '--results', required=True, help="dir to resulting annotations")
parser.add_argument('-na', '--no-animation',  default=True, help="no animation is shown.", action="store_true")
parser.add_argument('-p', '--plot', default=False, help="plot is shown.", action="store_true")
parser.add_argument('-q', '--quiet', help="minimalistic console output.", action="store_true")
parser.add_argument('-v', '--verbose', help="maximalistic console output.", action="store_true")
parser.add_argument('-i', '--ignore', nargs='+', type=str, help="ignore a list of classes.")
# argparse receiving list of classes with specific IoU (e.g., python main.py --set-class-iou person 0.7)
parser.add_argument('--set-class-iou', nargs='+', type=str, help="set IoU for a specific class.")
args = parser.parse_args()

'''
    0,0 ------> x (width)
     |
     |  (Left,Top)
     |      *_________
     |      |         |
            |         |
     y      |_________|
  (height)            *
                (Right,Bottom)
'''

if args.ignore is None:
    args.ignore = []

specific_iou_flagged = False
if args.set_class_iou is not None:
    specific_iou_flagged = True

GT_PATH = os.path.join(args.ground_truth)
DR_PATH = os.path.join(args.results)


"""
 throw error and exit
"""
def error(msg):
    print(msg)
    sys.exit(0)

"""
 check if the number is a float between 0.0 and 1.0
"""
def is_float_between_0_and_1(value):
    try:
        val = float(value)
        if val > 0.0 and val < 1.0:
            return True
        else:
            return False
    except ValueError:
        return False

"""
 Calculate the AP given the recall and precision array
    1st) We compute a version of the measured precision/recall curve with
         precision monotonically decreasing
    2nd) We compute the AP as the area under this curve by numerical integration.
"""
def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0)
    rec.append(1.0)
    mrec = rec[:]
    prec.insert(0, 0.0)
    prec.append(0.0)
    mpre = prec[:]
    """
     This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab: for i=numel(mpre)-1:-1:1
                    mpre(i)=max(mpre(i),mpre(i+1));
    """
    # matlab indexes start in 1 but python in 0, so I have to do:
    #     range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #     range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    """
     This part creates a list of indexes where the recall changes
        matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i) # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap, mrec, mpre


"""
 ground-truth
     Create a list of all the class names present in the ground-truth (gt_classes).
"""

gt_counter_per_class = collections.defaultdict(int)
counter_images_per_class = collections.defaultdict(int)

# Store the groundtruth in memory.  (Old was to write it out to json file)
groundtruth = {}

with jsonlines.open(GT_PATH) as reader:
    for obj in reader:
        bounding_boxes = []
        already_seen_classes = []

        # create ground-truth dictionary
        #class_name, left, top, right, bottom = line.split()
        frame_no = obj["frame_no"]
        for box in obj["boxes"]:
            class_name = box["class"]
            if class_name in args.ignore:
                continue
            confidence = box["conf"]
            left = box["x1"]
            right = box["x2"]
            top = box["y1"]
            bottom = box["y2"]
            # thebox list is left, top, right, bottom
            thebox = (left, top, right, bottom)
            bounding_boxes.append({"class_name":class_name, "bbox": thebox, "used":False, "confidence":confidence})
            # count that object
            gt_counter_per_class[class_name] += 1

            if class_name not in already_seen_classes:
                counter_images_per_class[class_name] += 1
                already_seen_classes.append(class_name)

        groundtruth[frame_no] = bounding_boxes
 

gt_classes = list(gt_counter_per_class.keys())
# sort the classes alphabetically
gt_classes = sorted(gt_classes)
n_classes = len(gt_classes)

dr_store = {}  # Used to be json
"""
 Check format of the flag --set-class-iou (if used)
    e.g. check if class exists
"""
if specific_iou_flagged:
    n_args = len(args.set_class_iou)
    error_msg = \
        '\n --set-class-iou [class_1] [IoU_1] [class_2] [IoU_2] [...]'
    if n_args % 2 != 0:
        error('Error, missing arguments. Flag usage:' + error_msg)
    # [class_1] [IoU_1] [class_2] [IoU_2]
    # specific_iou_classes = ['class_1', 'class_2']
    specific_iou_classes = args.set_class_iou[::2] # even
    # iou_list = ['IoU_1', 'IoU_2']
    iou_list = args.set_class_iou[1::2] # odd
    if len(specific_iou_classes) != len(iou_list):
        error('Error, missing arguments. Flag usage:' + error_msg)
    for tmp_class in specific_iou_classes:
        if tmp_class not in gt_classes:
                    error('Error, unknown class \"' + tmp_class + '\". Flag usage:' + error_msg)
    for num in iou_list:
        if not is_float_between_0_and_1(num):
            error('Error, IoU must be between 0.0 and 1.0. Flag usage:' + error_msg)

"""
 detection-results
"""
# get a list with the detection-results files
for class_index, class_name in enumerate(gt_classes):
    bounding_boxes = []
    with jsonlines.open(DR_PATH) as reader:
        for obj in reader:
            boxes = obj["boxes"]
            frame_no = obj["frame_no"]
            for box in boxes:
                tmp_class_name = box["class"]
                confidence = box["conf"]
                left = box["x1"]
                right = box["x2"]
                top = box["y1"]
                bottom = box["y2"]
                if tmp_class_name == class_name:
                    bounding_boxes.append({"confidence":confidence, "frame_no":frame_no, "bbox": [left, top, right, bottom]})
    # sort detection-results by decreasing confidence
    bounding_boxes.sort(key=lambda x:float(x['confidence']), reverse=True)
    dr_store[class_name] = bounding_boxes

"""
 Calculate the AP for each class
"""
sum_AP = 0.0
for class_index, class_name in enumerate(gt_classes):
    dr_data = dr_store[class_name] # Detection results

    """
        Assign detection-results to ground-truth objects
    """
    nd = len(dr_data)
    tp = [0] * nd # creates an array of zeros of size nd
    fp = [0] * nd
    for idx, detection in enumerate(dr_data):
        frame_no = detection["frame_no"]
        # assign detection-results to ground truth object if any
        # open ground-truth with that frame_no
        ground_truth_data = groundtruth[frame_no]
        ovmax = -1
        gt_match = -1
        # load detected object bounding-box
        bb = detection["bbox"]
        for obj in ground_truth_data:
            # look for a class_name match
            if obj["class_name"] == class_name:
                bbgt = obj["bbox"]
                bi = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
                iw = bi[2] - bi[0] + 1
                ih = bi[3] - bi[1] + 1
                if iw > 0 and ih > 0:
                    # compute overlap (IoU) = area of intersection / area of union
                    ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                    + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                    ov = iw * ih / ua
                    if ov > ovmax:
                        ovmax = ov
                        gt_match = obj

        # set minimum overlap
        min_overlap = MINOVERLAP
        if specific_iou_flagged:
            if class_name in specific_iou_classes:
                index = specific_iou_classes.index(class_name)
                min_overlap = float(iou_list[index])
        if ovmax >= min_overlap:
            if "difficult" not in gt_match:
                    if not bool(gt_match["used"]):
                        result_conf = detection["confidence"]
                        gt_conf = gt_match["confidence"]
                        if args.verbose:
                            print(f"conf_diff {gt_conf} {result_conf} {gt_conf - result_conf}")

                        # true positive
                        tp[idx] = 1
                        gt_match["used"] = True
                        groundtruth[frame_no] = ground_truth_data
                    else:
                        # false positive (multiple detection)
                        fp[idx] = 1
        else:
            # false positive
            fp[idx] = 1


    #print(tp)
    # compute precision/recall
    cumsum = 0
    for idx, val in enumerate(fp):
        fp[idx] += cumsum
        cumsum += val
    cumsum = 0
    for idx, val in enumerate(tp):
        tp[idx] += cumsum
        cumsum += val
    #print(tp)
    rec = tp[:]
    for idx, val in enumerate(tp):
        rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
    #print("Recall: ", rec)
    prec = tp[:]
    for idx, val in enumerate(tp):
        prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
    #print("Precision: ", prec)

    ap, mrec, mprec = voc_ap(rec[:], prec[:])
    sum_AP += ap


mAP = sum_AP / n_classes
text = "mAP = {0:.2f}%".format(mAP*100)
print(text)
