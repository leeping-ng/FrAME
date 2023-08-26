import datetime

import pandas as pd
from pathlib import Path
import pytorch_lightning as pl
import random


from utils import load_config, parse_args, save_predict_mode_results
from curve_fit import CurveFit
from ks_test import KSTest
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


if __name__ == "__main__":
    mode = parse_args()
    configs = load_config(CONFIG_PATH)

    # setup for saving results
    time_now = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    save_dir = configs["common"]["result_folder"] + "/" + time_now
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    pl.seed_everything(33, workers=True)
    model = Model.load_from_checkpoint(configs["common"]["bbsd_checkpoint_path"])
    trainer = pl.Trainer(enable_progress_bar=False, devices=1, num_nodes=1)

    source_data_module = DataModule(
        configs[mode]["source_images_dir"],
        configs[mode]["source_metadata_path"],
        configs["common"]["batch_size"],
    )
    ks_test = KSTest(configs["common"]["num_classes"], ALPHA)
    source_dataloader = source_data_module.predict_dataloader(PREPROCESS_TF)
    source_softmax = trainer.predict(model=model, dataloaders=source_dataloader)

    if mode == "predict":
        a = configs["predict"]["coefficients"]["a"]
        b = configs["predict"]["coefficients"]["b"]
        c = configs["predict"]["coefficients"]["c"]
        curve_fit = CurveFit(save_dir, a, b, c)
        target_data_module = DataModule(
            configs[mode]["target_images_dir"],
            configs[mode]["target_metadata_path"],
            configs["common"]["batch_size"],
        )
        target_dataloader = target_data_module.predict_dataloader(PREPROCESS_TF)
        target_softmax = trainer.predict(model=model, dataloaders=target_dataloader)

        ks_signal, ks_pvalue, shift_detected = ks_test.perform_test(
            source_softmax, target_softmax
        )

        # Else log function will be NAN
        estimated_perf_drop = None
        if c - ks_signal > EPSILON:
            estimated_perf_drop = curve_fit.perf_drop_from_signal(ks_signal)
        save_predict_mode_results(
            save_dir, estimated_perf_drop, ks_signal, ks_pvalue, shift_detected
        )

    elif mode == "fit":
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
            random.seed(0)
            target_dataloader = target_data_module.predict_dataloader(transform)

            target_softmax = trainer.predict(model=model, dataloaders=target_dataloader)
            target_roc = trainer.test(model=model, dataloaders=target_dataloader)[0][
                "test_roc-auc"
            ]

            ks_signal, ks_pvalue, shift_detected = ks_test.perform_test(
                source_softmax, target_softmax
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

        curve_fit.fit(df_tf)
        curve_fit.save_coefficients()
        curve_fit.plot_results(df_tf)
