import datetime
import pandas as pd
from pathlib import Path
import pytorch_lightning as pl
import torch
from typing import Any, Callable, Dict

from curve_fit import CurveFit
from dataloader import DataModule
from ks_test import KSTest
from model import Model
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
from utils import load_config, parse_args, save_predict_mode_results


CONFIG_PATH = "config.yml"
EPSILON = 0.01
ALPHA = 0.05


def predict_pipeline(
    configs: Dict[str, Any],
    save_dir: Path,
    model: Callable,
    source_softmax: torch.Tensor,
) -> None:
    """
    Pipeline for FRAME in Predict Mode, to predict the estimated performance drop.

    Args:
        configs (Dict[str, Any]): Configs stored in config.yml
        save_dir (Path): Directory to save results
        model (Callable): BBSD model
        source_softmax (torch.Tensor): Softmax tensor from source distribution
    """
    a = configs["predict"]["coefficients"]["a"]
    b = configs["predict"]["coefficients"]["b"]
    c = configs["predict"]["coefficients"]["c"]
    curve_fit = CurveFit(save_dir, a, b, c)

    # prepare data from target distribution
    target_data_module = DataModule(
        configs[mode]["target_images_dir"],
        configs[mode]["target_metadata_path"],
        configs["common"]["batch_size"],
    )
    target_dataloader = target_data_module.predict_dataloader(PREPROCESS_TF)
    target_softmax = trainer.predict(model=model, dataloaders=target_dataloader)

    # K-S test
    ks_signal, ks_pvalue, shift_detected = ks_test.perform_test(
        source_softmax, target_softmax
    )

    # estimate performance drop and save results
    estimated_perf_drop = None
    # Else log function will be NAN
    if c - ks_signal > EPSILON:
        estimated_perf_drop = curve_fit.perf_drop_from_signal(ks_signal)
    save_predict_mode_results(
        save_dir, estimated_perf_drop, ks_signal, ks_pvalue, shift_detected
    )


def fit_pipeline(
    configs: Dict[str, Any],
    save_dir: Path,
    model: Callable,
    source_softmax: torch.Tensor,
    source_dataloader: Callable,
) -> None:
    """
    Pipeline for FRAME in Fit Mode, to calculate the coefficients a, b, and c.

    Args:
        configs (Dict[str, Any]): Configs stored in config.yml
        save_dir (Path): Directory to save results
        model (Callable): BBSD model
        source_softmax (torch.Tensor): Softmax tensor from source distribution
        source_dataloader (Callable): Dataloader for source distribution
    """
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
    curve_fit = CurveFit(save_dir)

    # loop over different transforms
    for tf_name, transform in all_tf.items():
        # prepare data from target distribution
        target_dataloader = target_data_module.predict_dataloader(transform)
        target_softmax = trainer.predict(model=model, dataloaders=target_dataloader)
        target_roc = trainer.test(model=model, dataloaders=target_dataloader)[0][
            configs["fit"]["performance_metric"]
        ]
        perf_drop = 100 * (source_roc - target_roc) / source_roc

        # K-S test
        ks_signal, ks_pvalue, shift_detected = ks_test.perform_test(
            source_softmax, target_softmax
        )

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

    # Aggregate data once all transforms have been looped over
    print(df_tf)
    df_tf = df_tf.sort_values(by=["Actual perf drop"])
    df_tf.to_csv(
        save_dir / "fit_data.csv",
        index=False,
    )
    curve_fit.fit(df_tf)
    curve_fit.save_coefficients()
    curve_fit.plot_results(df_tf)


if __name__ == "__main__":
    mode = parse_args()
    configs = load_config(CONFIG_PATH)

    # preparation for saving results
    time_now = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    save_dir = Path(configs["common"]["result_folder"]) / time_now
    save_dir.mkdir(parents=True, exist_ok=True)

    # initialisation and setup
    pl.seed_everything(33, workers=True)
    model = Model.load_from_checkpoint(configs["common"]["bbsd_checkpoint_path"])
    trainer = pl.Trainer(enable_progress_bar=False, devices=1, num_nodes=1)
    ks_test = KSTest(configs["common"]["num_classes"], ALPHA)

    # prepare data from the source distribution
    source_data_module = DataModule(
        configs[mode]["source_images_dir"],
        configs[mode]["source_metadata_path"],
        configs["common"]["batch_size"],
    )
    source_dataloader = source_data_module.predict_dataloader(PREPROCESS_TF)
    source_softmax = trainer.predict(model=model, dataloaders=source_dataloader)

    if mode == "predict":
        predict_pipeline(configs, save_dir, model, source_softmax)
    elif mode == "fit":
        fit_pipeline(configs, save_dir, model, source_softmax, source_dataloader)
