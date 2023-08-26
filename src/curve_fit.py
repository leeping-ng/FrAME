import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import optimize
from typing import Union


class CurveFit:
    def __init__(
        self,
        save_dir: Path,
        a: Union[None, float] = None,
        b: Union[None, float] = None,
        c: Union[None, float] = None,
    ) -> None:
        """
        Curve fitting and creation for FRAME.

        Args:
            save_dir (Path): Directory to save results
            a (Union[None, float]): Coefficient a of the curve
            b (Union[None, float]): Coefficient b of the curve
            c (Union[None, float]): Coefficient c of the curve
        """
        self.save_dir = save_dir
        self.a = a
        self.b = b
        self.c = c

    def _exponential_fn(self, x: float, a: float, b: float, c: float) -> float:
        """
        Exponential function for non-linear least squares method to fit to. Note that coefficients
        a and b are intentionally fixed with negative values as this curve shape allows the
        optimiser to converge. Also note that to get signal from performance drop, use
        signal_from_perf_drop() method with correct values of a, b, and c.

        Args:
            x (float): Variable x, representing performance drop
            a (float): Coefficient a of the curve
            b (float): Coefficient b of the curve
            c (float): Coefficient c of the curve

        Returns:
            float: Variable representing K-S signal
        """
        return -a * np.exp(-b * x) + c

    def signal_from_perf_drop(self, perf_drop: float) -> float:
        """
        Given the percentage performance drop, calculate the K-S signal from the curve equation:
        y = a*e^(bx) + c

        Args:
            perf_drop (float): Performance drop in %

        Returns:
            float: K-S signal
        """
        return self.a * np.exp(self.b * perf_drop) + self.c

    def perf_drop_from_signal(self, signal: float) -> float:
        """
        Given the K-S signal, calculate the percentage performance drop from the curve equation:
        x = (1/b)*ln((y-c)/a)

        Args:
            signal (float): K-S signal

        Returns:
            float: Performance drop in %
        """
        return (1 / self.b) * np.log((signal - self.c) / self.a)

    def fit(self, df_tf: pd.DataFrame) -> None:
        """
        Fit data to a curve using non-linear least squares method, printing the coefficients
        of the curve to terminal.

        Args:
            df_tf (pd.DataFrame): DataFrame containing information on performance drop and K-S
                signals.
        """
        popt, _ = optimize.curve_fit(
            self._exponential_fn, df_tf["Actual perf drop"], df_tf["K-S signal"]
        )
        # intentionally negate a and b to reverse negation done in _exponential_fn() for fitting
        self.a, self.b, self.c = -popt[0], -popt[1], popt[2]
        print("COEFFICIENTS")
        print("a:", self.a)
        print("b:", self.b)
        print("c:", self.c)

    def save_coefficients(self) -> None:
        """
        Saves coefficients to a text file.
        """
        filename = (Path(self.save_dir) / "coefficients.txt").resolve()
        with open(filename, "w") as f:
            f.write("a: " + str(self.a) + "\n")
            f.write("b: " + str(self.b) + "\n")
            f.write("c: " + str(self.c) + "\n")

    def plot_results(self, df_tf: pd.DataFrame) -> None:
        """
        Plot results from dataframe and curve fit, and save to a figure.

        Args:
            df_tf (pd.DataFrame): DataFrame containing information on performance drop and K-S
                signals.
        """
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
