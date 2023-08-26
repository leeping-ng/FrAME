import argparse
from pathlib import Path
import yaml


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
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
    save_dir, estimated_perf_drop, ks_signal, ks_pvalue, shift_detected
):
    filename = (Path(save_dir) / "estimated_performance.txt").resolve()
    with open(filename, "w") as f:
        f.write(f"Estimated performance drop: {estimated_perf_drop:.2f}%\n")
        f.write(f"K-S signal: {ks_signal:.4f}\n")
        f.write(f"P-value: {ks_pvalue:.4f}\n")
        f.write(f"Shift detected: {str(shift_detected)}")

    print(f"Estimated performance drop: {estimated_perf_drop:.2f}%")
    print(f"K-S signal: {ks_signal:.4f}")
    print(f"P-value: {ks_pvalue:.4f}")
    print(f"Shift detected: {str(shift_detected)}")
