import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style("white")
sns.set_palette("husl")

def analyze(filename):
    diffs = []
    num_positive = 0.
    num_negative = 0.
    num_zero = 0.
    with open(filename) as f:
        for line in f:
            if "conf_diff" not in line:
                continue
            vals = line.rstrip().split(" ")
            diff = round(float(vals[3]), 3)
            diffs.append(diff)
            if diff == 0:
                num_zero += 1
            elif diff > 0:
                num_positive += 1
            else:
                num_negative += 1

    plt.hist(diffs, 60)
    median = np.median(diffs)
    average = np.average(diffs)
    print(f"Max: {max(diffs)}")
    print(f"Median: {median}")
    print(f"Average: {average}")
    print(f"% positive: {num_positive * 100 / len(diffs)}")
    print(f"% zero: {num_zero * 100 / len(diffs)}")
    print(f"% negative: {num_negative * 100 / len(diffs)}")
    plt.xlabel("Ground truth conf - Result conf")
    plt.ylabel("Histogram")
    plt.savefig("/tmp/conf-diffs-hist.pdf")

if __name__ == "__main__":
    filename = sys.argv[1]
    analyze(filename)
