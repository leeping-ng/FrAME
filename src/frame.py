import argparse
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import pytorch_lightning as pl
import random
from scipy import stats, optimize

from config import load_config
from model import Model
from dataloader import DataModule
from transforms import (
    PREPROCESS_TF,
    UNCHANGED_TF,
    BLUR_TF,
    SHARPEN_TF,
    SALT_PEPPER_NOISE_TF,
    SPECKLE_NOISE_TF,
    CONTRAST_INC_TF,
    CONTRAST_DEC_TF,
    GAMMA_INC_TF,
    GAMMA_DEC_TF,
    MAGNIFY_TF,
)

CONFIG_PATH = "config.yml"
EPSILON = 0.01
ALPHA = 0.05


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


def extract_softmaxes_by_class(source_softmax, target_softmax, num_classes):
    # initialise empty data structures to be extended over batches
    source_softmax_by_class = {}
    target_softmax_by_class = {}

    for i in range(num_classes):
        source_softmax_by_class[i] = []
        target_softmax_by_class[i] = []

    # loop over batches to aggregate softmax
    for i, (source_batch, target_batch) in enumerate(
        zip(source_softmax, target_softmax)
    ):
        source_batch = source_batch.numpy()
        target_batch = target_batch.numpy()

        for j in range(num_classes):
            source_softmax_by_class[j].extend(list(source_batch[:, j].squeeze()))
            target_softmax_by_class[j].extend(list(target_batch[:, j].squeeze()))

    return source_softmax_by_class, target_softmax_by_class


def multiple_univariate_ks_test(source_softmax_by_class, target_softmax_by_class):
    # K-S taking in softmax
    ks_result = {}
    shift_detected = False
    ks_signal = 0
    ks_pvalue = 0
    for i in range(num_classes):
        ks_result[i] = stats.ks_2samp(
            source_softmax_by_class[i], target_softmax_by_class[i]
        )
        # Reject null hypothesis if any p-value < Bonferroni corrected significance level
        if ks_result[i].pvalue < ALPHA / num_classes:
            shift_detected = True

        ks_signal += ks_result[i].statistic
        ks_pvalue += ks_result[i].pvalue
    ks_signal /= num_classes
    ks_pvalue /= num_classes

    return ks_signal, ks_pvalue, shift_detected


def exponential_func(x, a, b, c):
    # fix curve shape such that non-linear least squares is able to converge
    return -a * np.exp(-b * x) + c


def perf_drop_from_signal(signal, a, b, c):
    """
    y = a*e^(bx) + c
    x = (1/b)*ln((y-c)/a)
    Where a, b, c = *popt
    """
    return (1 / b) * np.log((signal - c) / a)


def fit_and_save_coefficients(df_tf, save_dir):
    popt, _ = optimize.curve_fit(
        exponential_func, df_tf["Actual perf drop"], df_tf["K-S signal"]
    )
    a, b, c = -popt[0], -popt[1], popt[2]
    print("COEFFICIENTS")
    print("a:", a)
    print("b:", b)
    print("c:", c)
    filename = save_dir + "/coefficients.txt"
    with open(filename, "w") as f:
        f.write("a: " + str(a) + "\n")
        f.write("b: " + str(b) + "\n")
        f.write("c: " + str(c) + "\n")

    return a, b, c


def plot_results(df_tf, save_dir, a, b, c):
    plt.figure()
    plt.scatter(df_tf["Actual perf drop"], df_tf["K-S signal"], s=10)
    # negative a & b to undo negative in exponential_func()
    plt.plot(
        df_tf["Actual perf drop"],
        exponential_func(df_tf["Actual perf drop"], -a, -b, c),
        "r-",
        linewidth=3,
    )
    plt.axis(xmin=0, xmax=40, ymin=0, ymax=0.85)
    plt.ylabel("K-S Signal")
    plt.xlabel("% Drop in Performance")
    plt.legend(["Transform Severity\nResult", "Fitted Exponential\nCurve"])
    plt.title("FRAME in Fit Mode")
    plt.savefig(save_dir + "/" + mode + "_plot.jpg")


if __name__ == "__main__":
    mode = parse_args()
    configs = load_config(CONFIG_PATH)

    num_classes = configs["common"]["num_classes"]

    pl.seed_everything(33, workers=True)
    model = Model.load_from_checkpoint(configs["common"]["bbsd_checkpoint_path"])
    trainer = pl.Trainer(enable_progress_bar=False, devices=1, num_nodes=1)

    source_data_module = DataModule(
        configs[mode]["source_images_dir"],
        configs[mode]["source_metadata_path"],
        configs["common"]["batch_size"],
    )
    source_dataloader = source_data_module.predict_dataloader(PREPROCESS_TF)
    source_softmax = trainer.predict(model=model, dataloaders=source_dataloader)

    if mode == "predict":
        a = configs["predict"]["coefficients"]["a"]
        b = configs["predict"]["coefficients"]["b"]
        c = configs["predict"]["coefficients"]["c"]
        target_data_module = DataModule(
            configs[mode]["target_images_dir"],
            configs[mode]["target_metadata_path"],
            configs["common"]["batch_size"],
        )
        target_dataloader = target_data_module.predict_dataloader(PREPROCESS_TF)
        target_softmax = trainer.predict(model=model, dataloaders=target_dataloader)

        source_softmax_by_class, target_softmax_by_class = extract_softmaxes_by_class(
            source_softmax, target_softmax, num_classes
        )

        ks_signal, ks_pvalue, shift_detected = multiple_univariate_ks_test(
            source_softmax_by_class, target_softmax_by_class
        )

        # Else log function will be NAN
        estimated_perf_drop = None
        if c - ks_signal > 0.01:
            estimated_perf_drop = perf_drop_from_signal(ks_signal, a, b, c)

        print("Estimated performance drop: ", estimated_perf_drop, "%")

    elif mode == "fit":
        # setup for saving results
        time_now = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
        save_dir = configs["common"]["result_folder"] + "/" + time_now
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        df_tf = pd.DataFrame(
            columns=[
                "Transform",
                "Source ROC-AUC",
                "Target ROC-AUC",
                "Actual perf drop",
                "K-S signal",
                "K-S p-value",
                "Shift detected",
            ]
        )

        # all_tf = (
        #     UNCHANGED_TF
        #     | BLUR_TF
        #     | SHARPEN_TF
        #     | SALT_PEPPER_NOISE_TF
        #     | SPECKLE_NOISE_TF
        #     | CONTRAST_INC_TF
        #     | CONTRAST_DEC_TF
        #     | GAMMA_INC_TF
        #     | GAMMA_DEC_TF
        #     | MAGNIFY_TF
        # )
        all_tf = UNCHANGED_TF | SHARPEN_TF

        source_roc = trainer.test(model=model, dataloaders=source_dataloader)[0][
            "test_roc-auc"
        ]
        target_data_module = DataModule(
            configs[mode]["source_images_dir"],
            configs[mode]["source_metadata_path"],
            configs["common"]["batch_size"],
        )
        # loop over different transforms
        for tf_name, transform in all_tf.items():
            random.seed(0)
            target_dataloader = target_data_module.predict_dataloader(transform)

            target_softmax = trainer.predict(model=model, dataloaders=target_dataloader)
            target_roc = trainer.test(model=model, dataloaders=target_dataloader)[0][
                "test_roc-auc"
            ]

            (
                source_softmax_by_class,
                target_softmax_by_class,
            ) = extract_softmaxes_by_class(source_softmax, target_softmax, num_classes)

            ks_signal, ks_pvalue, shift_detected = multiple_univariate_ks_test(
                source_softmax_by_class, target_softmax_by_class
            )

            perf_drop = 100 * (source_roc - target_roc) / source_roc

            res_tf = {
                "Transform": tf_name,
                "Source ROC-AUC": source_roc,
                "Target ROC-AUC": target_roc,
                "Actual perf drop": perf_drop,
                "K-S signal": ks_signal,
                "K-S p-value": ks_pvalue,
                "Shift detected": shift_detected,
            }

            df_tf = pd.concat([df_tf, pd.DataFrame([res_tf])], ignore_index=True)

        print(df_tf)
        df_tf = df_tf.sort_values(by=["Actual perf drop"])

        df_tf.to_csv(
            save_dir + "/" + mode + "_data.csv",
            index=False,
        )

        a, b, c = fit_and_save_coefficients(df_tf, save_dir)
        plot_results(df_tf, save_dir, a, b, c)
