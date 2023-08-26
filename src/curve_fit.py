import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import optimize


class CurveFit:
    def __init__(self, save_dir, a=None, b=None, c=None):
        self.save_dir = save_dir
        self.a = a
        self.b = b
        self.c = c

    def _exponential_fn(self, x, a, b, c):
        # fix curve shape such that non-linear least squares is able to converge
        # only using for optimise curve fit
        return -a * np.exp(-b * x) + c

    def signal_from_perf_drop(self, perf_drop):
        return self.a * np.exp(self.b * perf_drop) + self.c

    def perf_drop_from_signal(self, signal):
        """
        y = a*e^(bx) + c
        x = (1/b)*ln((y-c)/a)
        """
        return (1 / self.b) * np.log((signal - self.c) / self.a)

    def fit(self, df_tf):
        popt, _ = optimize.curve_fit(
            self._exponential_fn, df_tf["Actual perf drop"], df_tf["K-S signal"]
        )
        # convert negative values back
        self.a, self.b, self.c = -popt[0], -popt[1], popt[2]
        print("COEFFICIENTS")
        print("a:", self.a)
        print("b:", self.b)
        print("c:", self.c)

    def save_coefficients(self):
        filename = (Path(self.save_dir) / "coefficients.txt").resolve()
        with open(filename, "w") as f:
            f.write("a: " + str(self.a) + "\n")
            f.write("b: " + str(self.b) + "\n")
            f.write("c: " + str(self.c) + "\n")

    def plot_results(self, df_tf):
        plt.figure()
        plt.scatter(df_tf["Actual perf drop"], df_tf["K-S signal"], s=10)
        plt.plot(
            df_tf["Actual perf drop"],
            self.signal_from_perf_drop(df_tf["Actual perf drop"]),
            "r-",
            linewidth=3,
        )
        plt.axis(xmin=0, xmax=40, ymin=0, ymax=0.85)
        plt.ylabel("K-S Signal")
        plt.xlabel("% Drop in Performance")
        plt.legend(["Transform Severity\nResult", "Fitted Exponential\nCurve"])
        plt.title("FRAME in Fit Mode")
        filename = (Path(self.save_dir) / "plot.jpg").resolve()
        plt.savefig(filename)
