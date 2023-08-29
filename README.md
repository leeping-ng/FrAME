# FRAME: FRamework for AI Monitoring & Evaluation

This repository contains sample code for my Master's thesis **"Performance Monitoring of Medical Imaging AI"** at Imperial College London. For this project, I developed **FRAME** (**FR**amework for **A**I **M**onitoring & **E**valuation), a framework for performance monitoring which can detect shifts in data distribution and estimate the performance drop of a model in production.

FRAME is designed to be applicable in challenging real-world scenarios, such as in the post-deployment phase, and in situations where **ground truth labels are unavailable**, samples are limited, and class distributions are unknown for the real-world data.

While FRAME can be applied to any computer vision task, this software implementation is specific to the following conditions:
- **Task**: Image classification
- **Dataset**: X-ray images in grayscale with a single channel
- **Model**: Model code in Pytorch and weights in `.ckpt` format

If your project meets these requirements, feel free to use FRAME by following the setup instructions below. If you run into any issues or errors, please raise an [issue](https://github.com/leeping-ng/frame/issues) and I'll do my best to address it.


## How FRAME Works

We define 2 different dataset distributions:
- **Source distribution**: Distribution of original data that model was trained on
- **Target distribution**: Distribution of real-world data that model performs inference on

It is not uncommon for an AI model’s performance to deteriorate over time after deployment. One of the reasons is data distribution shift, where the source and target distributions are no longer equal, and examples of root causes include changes in patient demographics or the medical imaging device over time. FRAME can be used to estimate the performance of the model on samples from the *target* distribution, without requiring ground truth labels from the *target* distribution.

Firstly, FRAME is fit on images and labels from the *source* distribution, and 85 different transforms are applied on these to simulate images from the *target* distribution. This pipeline is shown in the subsequent [FRAME in Fit Mode](#frame-in-fit-mode) section. This establishes a relationship between shift detection signals and % performance drop.

Next, using the established relationship, FRAME is used to make predictions on real-world samples from the *target* distribution. This pipeline is shown in the subsequent [FRAME in Predict Mode](#frame-in-predict-mode) section. The status of whether a shift has occurred and the estimated performance drop of the model will be obtained. Users can then decide if corrective actions such as model re-training is required.


## Setup
1. Install the required dependencies:
    ```
    > pip install -r requirements.txt
    ```
2. Prepare data from the *source* distribution and update the following fields in [config.yml](src/config.yml) accordingly:
    - `source_images_dir`: Store images from *source* distribution here
    - `source_metadata_path`: Create a *csv* file with 2 column headers `image` and `label`. Populate the `image` column with paths of all images in `source_images_dir`. Populate the `label` column with corresponding ground truth labels as integers.
3. If data from the *target* distribution is available, prepare the data and update the following fields in [config.yml](src/config.yml):
    - `target_images_dir`: Store images from *target* distribution here
    - `target_metadata_path`: Create a *csv* file with 2 column headers `image` and `label`. Populate the `image` column with paths of all images in `target_images_dir`. The `label` column can be left empty.
3. Prepare the BBSD model as an input to FRAME:
    - Update in [config.yml](src/config.yml) under the `common` field:
        - `num_classes`: Number of classes for classification task
        - `batch_size`: Batch size for inference
        - `bbsd_checkpoint_path`: Path of stored checkpoint for PyTorch model with `.ckpt` extension
    - Migrate your PyTorch model code into [model.py](src/model.py) as a Pytorch Lightning module. This module will be imported by [frame.py](src/frame.py), which is the main Python script. For reference, see [model_sample.py](src/model_sample.py) which uses a ResNet-18 model.
        - The `predict_step()` method should return a softmax vector as a PyTorch tensor
        - The `test_step()` method should return the chosen evaluation metric (e.g. ROC-AUC) as a float
  


## FRAME in Fit Mode

In this scenario, the user is close to deploying or has just deployed their model, has not seen any real-world data (from target distribution) yet, and wants to monitor performance in the future. FRAME will fit an exponential function of the form *y=ae^bx+c* on the data, relating the shift detection signal (from the K-S test) to the % performance drop.

!["Fit Mode"](images/frame_fit_pipeline.png)

To use FRAME in Fit Mode, in [config.yml](src/config.yml), update the field for `performance_metric` under `fit` with the name of the evaluation metric that you are interested in and was used when you trained the model, such as *test_roc-auc*. Then, run the following commands:

```
> cd src
> python frame.py --mode fit
```

FRAME will produce coefficients a, b, and c which define the relationship between the shift detection signal and performance drop.


## FRAME in Predict Mode

In this scenario, the user has deployed their model some time ago, has collected real-world data (from target distribution), and wants to estimate the model’s drop in performance (if any) to assess if corrective actions are needed. Prior to this, the user would have used Fit Mode to calculate the coefficients a, b, and c.

!["Predict Mode"](images/frame_predict_pipeline.png)

To use FRAME in Predict Mode, in [config.yml](src/config.yml), update the fields for coefficients `a`, `b`, `c` under `predict` and `coefficients`. Then, run the following commands:

```
> cd src
> python frame.py --mode predict
```

FRAME will predict the estimated percentage drop of `performance_metric`, and also return a Boolean to indicate if a shift has been detected.