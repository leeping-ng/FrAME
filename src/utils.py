import argparse
from pathlib import Path
from typing import Any, Dict
import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads data from config file into dictionary.

    Args:
        config_path (str): Path of config file

    Returns:
        configs (Dict[str, Any]): Configs as a dictionary
    """

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def parse_args() -> str:
    """
    Parse command line arguments to choose either "fit" or "predict" mode.

    Returns:
        args.mode (str): Either "fit" or "predict" strings
    """

    parser = argparse.ArgumentParser("FRAME")
    parser.add_argument(
        "--mode",
        choices=["fit", "predict"],
        required=True,
        help="Choose between 'fit' or 'predict' modes",
    )
    args = parser.parse_args()
    return args.mode


def save_predict_mode_results(
    save_dir: Path,
    estimated_perf_drop: float,
    ks_signal: float,
    ks_pvalue: float,
    shift_detected: bool,
) -> None:
    """
    Helper function for Predict Mode to save results and print to screen.

    Args:
        save_dir (Path): Directory to save results
        estimated_perf_drop (float): Estimated percentage drop in performance
        ks_signal (float): Signal from K-S test
        ks_pvalue (float): P-value from K-S test
        shift_detected (bool): Whether a shift was detected
    """

    filename = (save_dir / "estimated_performance.txt").resolve()
    with open(filename, "w") as f:
        f.write(f"Estimated performance drop: {estimated_perf_drop:.2f}%\n")
        f.write(f"K-S signal: {ks_signal:.4f}\n")
        f.write(f"P-value: {ks_pvalue:.4f}\n")
        f.write(f"Shift detected: {str(shift_detected)}")

    print(f"Estimated performance drop: {estimated_perf_drop:.2f}%")
    print(f"K-S signal: {ks_signal:.4f}")
    print(f"P-value: {ks_pvalue:.4f}")
    print(f"Shift detected: {str(shift_detected)}")
