from skimage.util import random_noise
import torch
from torchvision import transforms


class ContrastTransform:
    def __init__(self, contrast_factor: float) -> None:
        """
        Adjust the contrast of an image.

        Args:
            contrast_factor (float): A contrast factor of 1 gives the original image, while
            <1 decreases the contrast and >1 increases the contrast.
        """
        self.contrast_factor = contrast_factor

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply the contrast transformation.

        Args:
            tensor (torch.Tensor): Tensor of original image

        Returns:
            tensor (torch.Tensor): Tensor of transformed image
        """

        tensor = transforms.functional.adjust_contrast(tensor, self.contrast_factor)
        return tensor


class GammaTransform:
    def __init__(self, gamma_factor: float) -> None:
        """
        Adjust the gamma of an image.

        Args:
            gamma_factor (float): A gamma factor of 1 gives the original image, while
            <1 decreases the gamma and >1 increases the gamma.
        """
        self.gamma_factor = gamma_factor

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply the gamma correction.

        Args:
            tensor (torch.Tensor): Tensor of original image

        Returns:
            tensor (torch.Tensor): Tensor of transformed image
        """
        tensor = transforms.functional.adjust_gamma(tensor, self.gamma_factor)
        return tensor


class SaltPepperNoiseTransform:
    def __init__(self, amount: float) -> None:
        """
        Add salt and pepper noise to an image.

        Args:
            amount (float): Proportion of image pixels to replace with noise (between 0 and 1)
        """
        self.amount = amount

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply the salt and pepper noise.

        Args:
            tensor (torch.Tensor): Tensor of original image

        Returns:
            tensor (torch.Tensor): Tensor of transformed image
        """
        tensor = tensor[0, :, :]
        tensor = torch.tensor(
            random_noise(
                tensor, mode="s&p", salt_vs_pepper=0.5, clip=True, amount=self.amount
            )
        )
        tensor = tensor.expand(3, 224, 224)
        return tensor


class SpeckleNoiseTransform:
    def __init__(self, variance: float) -> None:
        """
        Add speckle noise to an image.

        Args:
            variance (float): Variance of random distribution. The higher the variance, the
            greater the noise.
        """
        self.variance = variance

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply the speckle noise.

        Args:
            tensor (torch.Tensor): Tensor of original image

        Returns:
            tensor (torch.Tensor): Tensor of transformed image
        """
        tensor = tensor[0, :, :]
        tensor = torch.tensor(
            random_noise(tensor, mode="speckle", clip=True, var=self.variance)
        )
        tensor = tensor.expand(3, 224, 224).float()
        return tensor


class BlurTransform:
    def __init__(self, kernel_size: float) -> None:
        """
        Add blurring effect to an image.

        Args:
            kernel_size (float): Size of Gaussian blur kernel, the larger the size, the greater
            the blur effect
        """
        self.kernel_size = kernel_size

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply the blur transformation.

        Args:
            tensor (torch.Tensor): Tensor of original image

        Returns:
            tensor (torch.Tensor): Tensor of transformed image
        """
        tensor = transforms.functional.gaussian_blur(tensor, self.kernel_size)
        return tensor


class SharpenTransform:
    def __init__(self, sharpness_factor: float) -> None:
        """
        Adjust the sharpness of an image.

        Args:
            sharpness_factor (float): A sharpness factor of 1 gives the original image, while
            <1 decreases the sharpness and >1 increases the sharpness.
        """
        self.sharpness_factor = sharpness_factor

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply the sharpen transformation.

        Args:
            tensor (torch.Tensor): Tensor of original image

        Returns:
            tensor (torch.Tensor): Tensor of transformed image
        """
        tensor = transforms.functional.adjust_sharpness(tensor, self.sharpness_factor)
        return tensor


PREPROCESS_TF = transforms.Compose([transforms.ToTensor()])

UNCHANGED_TF = {"Unchanged": transforms.Compose([PREPROCESS_TF])}


BLUR_TF = {
    "Blur 7": transforms.Compose([PREPROCESS_TF, BlurTransform(7)]),
    "Blur 11": transforms.Compose([PREPROCESS_TF, BlurTransform(11)]),
    "Blur 17": transforms.Compose([PREPROCESS_TF, BlurTransform(17)]),
    "Blur 23": transforms.Compose([PREPROCESS_TF, BlurTransform(23)]),
    "Blur 37": transforms.Compose([PREPROCESS_TF, BlurTransform(37)]),
    "Blur 53": transforms.Compose([PREPROCESS_TF, BlurTransform(53)]),
    "Blur 97": transforms.Compose([PREPROCESS_TF, BlurTransform(97)]),
}


SHARPEN_TF = {
    "Sharpen 10": transforms.Compose([PREPROCESS_TF, SharpenTransform(10)]),
    "Sharpen 17": transforms.Compose([PREPROCESS_TF, SharpenTransform(17)]),
    "Sharpen 20": transforms.Compose([PREPROCESS_TF, SharpenTransform(20)]),
    "Sharpen 24": transforms.Compose([PREPROCESS_TF, SharpenTransform(24)]),
    "Sharpen 38": transforms.Compose([PREPROCESS_TF, SharpenTransform(38)]),
    "Sharpen 46": transforms.Compose([PREPROCESS_TF, SharpenTransform(46)]),
    "Sharpen 58": transforms.Compose([PREPROCESS_TF, SharpenTransform(58)]),
    "Sharpen 80": transforms.Compose([PREPROCESS_TF, SharpenTransform(80)]),
}


SALT_PEPPER_NOISE_TF = {
    "Salt Pepper Noise 0.001": transforms.Compose(
        [PREPROCESS_TF, SaltPepperNoiseTransform(0.001)]
    ),
    "Salt Pepper Noise 0.0016": transforms.Compose(
        [PREPROCESS_TF, SaltPepperNoiseTransform(0.0016)]
    ),
    "Salt Pepper Noise 0.0022": transforms.Compose(
        [PREPROCESS_TF, SaltPepperNoiseTransform(0.0022)]
    ),
    "Salt Pepper Noise 0.003": transforms.Compose(
        [PREPROCESS_TF, SaltPepperNoiseTransform(0.003)]
    ),
    "Salt Pepper Noise 0.007": transforms.Compose(
        [PREPROCESS_TF, SaltPepperNoiseTransform(0.007)]
    ),
    "Salt Pepper Noise 0.012": transforms.Compose(
        [PREPROCESS_TF, SaltPepperNoiseTransform(0.012)]
    ),
    "Salt Pepper Noise 0.016": transforms.Compose(
        [PREPROCESS_TF, SaltPepperNoiseTransform(0.016)]
    ),
    "Salt Pepper Noise 0.025": transforms.Compose(
        [PREPROCESS_TF, SaltPepperNoiseTransform(0.025)]
    ),
    "Salt Pepper Noise 0.056": transforms.Compose(
        [PREPROCESS_TF, SaltPepperNoiseTransform(0.056)]
    ),
    "Salt Pepper Noise 0.066": transforms.Compose(
        [PREPROCESS_TF, SaltPepperNoiseTransform(0.066)]
    ),
}


SPECKLE_NOISE_TF = {
    "Speckle Noise 0.0006": transforms.Compose(
        [PREPROCESS_TF, SpeckleNoiseTransform(0.0006)]
    ),
    "Speckle Noise 0.0014": transforms.Compose(
        [PREPROCESS_TF, SpeckleNoiseTransform(0.0014)]
    ),
    "Speckle Noise 0.0026": transforms.Compose(
        [PREPROCESS_TF, SpeckleNoiseTransform(0.0026)]
    ),
    "Speckle Noise 0.004": transforms.Compose(
        [PREPROCESS_TF, SpeckleNoiseTransform(0.004)]
    ),
    "Speckle Noise 0.01": transforms.Compose(
        [PREPROCESS_TF, SpeckleNoiseTransform(0.01)]
    ),
    "Speckle Noise 0.02": transforms.Compose(
        [PREPROCESS_TF, SpeckleNoiseTransform(0.02)]
    ),
    "Speckle Noise 0.035": transforms.Compose(
        [PREPROCESS_TF, SpeckleNoiseTransform(0.035)]
    ),
    "Speckle Noise 0.065": transforms.Compose(
        [PREPROCESS_TF, SpeckleNoiseTransform(0.065)]
    ),
    "Speckle Noise 2": transforms.Compose([PREPROCESS_TF, SpeckleNoiseTransform(2)]),
    "Speckle Noise 4": transforms.Compose([PREPROCESS_TF, SpeckleNoiseTransform(4)]),
}


CONTRAST_INC_TF = {
    "Contrast Inc 2": transforms.Compose([PREPROCESS_TF, ContrastTransform(2)]),
    "Contrast Inc 2.25": transforms.Compose([PREPROCESS_TF, ContrastTransform(2.25)]),
    "Contrast Inc 2.45": transforms.Compose([PREPROCESS_TF, ContrastTransform(2.45)]),
    "Contrast Inc 2.8": transforms.Compose([PREPROCESS_TF, ContrastTransform(2.8)]),
    "Contrast Inc 4.0": transforms.Compose([PREPROCESS_TF, ContrastTransform(4.0)]),
    "Contrast Inc 4.9": transforms.Compose([PREPROCESS_TF, ContrastTransform(4.9)]),
    "Contrast Inc 6.0": transforms.Compose([PREPROCESS_TF, ContrastTransform(6.0)]),
    "Contrast Inc 8.5": transforms.Compose([PREPROCESS_TF, ContrastTransform(8.5)]),
    "Contrast Inc 22": transforms.Compose([PREPROCESS_TF, ContrastTransform(22)]),
    "Contrast Inc 50": transforms.Compose([PREPROCESS_TF, ContrastTransform(50)]),
}


CONTRAST_DEC_TF = {
    "Contrast Dec 0.32": transforms.Compose([PREPROCESS_TF, ContrastTransform(0.32)]),
    "Contrast Dec 0.26": transforms.Compose([PREPROCESS_TF, ContrastTransform(0.26)]),
    "Contrast Dec 0.23": transforms.Compose([PREPROCESS_TF, ContrastTransform(0.23)]),
    "Contrast Dec 0.21": transforms.Compose([PREPROCESS_TF, ContrastTransform(0.21)]),
    "Contrast Dec 0.155": transforms.Compose([PREPROCESS_TF, ContrastTransform(0.155)]),
    "Contrast Dec 0.135": transforms.Compose([PREPROCESS_TF, ContrastTransform(0.135)]),
    "Contrast Dec 0.115": transforms.Compose([PREPROCESS_TF, ContrastTransform(0.115)]),
    "Contrast Dec 0.10": transforms.Compose([PREPROCESS_TF, ContrastTransform(0.10)]),
    "Contrast Dec 0.09": transforms.Compose([PREPROCESS_TF, ContrastTransform(0.09)]),
    "Contrast Dec 0.075": transforms.Compose([PREPROCESS_TF, ContrastTransform(0.075)]),
}

GAMMA_INC_TF = {
    "Gamma Inc 2.1": transforms.Compose([PREPROCESS_TF, GammaTransform(2.1)]),
    "Gamma Inc 2.4": transforms.Compose([PREPROCESS_TF, GammaTransform(2.4)]),
    "Gamma Inc 2.7": transforms.Compose([PREPROCESS_TF, GammaTransform(2.7)]),
    "Gamma Inc 2.9": transforms.Compose([PREPROCESS_TF, GammaTransform(2.9)]),
    "Gamma Inc 4.0": transforms.Compose([PREPROCESS_TF, GammaTransform(4.0)]),
    "Gamma Inc 5.2": transforms.Compose([PREPROCESS_TF, GammaTransform(5.2)]),
    "Gamma Inc 6.4": transforms.Compose([PREPROCESS_TF, GammaTransform(6.4)]),
    "Gamma Inc 7.4": transforms.Compose([PREPROCESS_TF, GammaTransform(7.4)]),
    "Gamma Inc 8.2": transforms.Compose([PREPROCESS_TF, GammaTransform(8.2)]),
    "Gamma Inc 8.8": transforms.Compose([PREPROCESS_TF, GammaTransform(8.8)]),
}


GAMMA_DEC_TF = {
    "Gamma Dec 0.37": transforms.Compose([PREPROCESS_TF, GammaTransform(0.37)]),
    "Gamma Dec 0.31": transforms.Compose([PREPROCESS_TF, GammaTransform(0.31)]),
    "Gamma Dec 0.28": transforms.Compose([PREPROCESS_TF, GammaTransform(0.28)]),
    "Gamma Dec 0.26": transforms.Compose([PREPROCESS_TF, GammaTransform(0.26)]),
    "Gamma Dec 0.21": transforms.Compose([PREPROCESS_TF, GammaTransform(0.21)]),
    "Gamma Dec 0.185": transforms.Compose([PREPROCESS_TF, GammaTransform(0.185)]),
    "Gamma Dec 0.16": transforms.Compose([PREPROCESS_TF, GammaTransform(0.16)]),
    "Gamma Dec 0.14": transforms.Compose([PREPROCESS_TF, GammaTransform(0.14)]),
    "Gamma Dec 0.12": transforms.Compose([PREPROCESS_TF, GammaTransform(0.12)]),
    "Gamma Dec 0.10": transforms.Compose([PREPROCESS_TF, GammaTransform(0.10)]),
}


MAGNIFY_TF = {
    "Magnify 1.44": transforms.Compose(
        [
            PREPROCESS_TF,
            transforms.Resize(int(224 * 1.44), antialias=True),
            transforms.CenterCrop(224),
        ]
    ),
    "Magnify 1.5": transforms.Compose(
        [
            PREPROCESS_TF,
            transforms.Resize(int(224 * 1.5), antialias=True),
            transforms.CenterCrop(224),
        ]
    ),
    "Magnify 1.56": transforms.Compose(
        [
            PREPROCESS_TF,
            transforms.Resize(int(224 * 1.56), antialias=True),
            transforms.CenterCrop(224),
        ]
    ),
    "Magnify 1.6": transforms.Compose(
        [
            PREPROCESS_TF,
            transforms.Resize(int(224 * 1.6), antialias=True),
            transforms.CenterCrop(224),
        ]
    ),
    "Magnify 1.7": transforms.Compose(
        [
            PREPROCESS_TF,
            transforms.Resize(int(224 * 1.7), antialias=True),
            transforms.CenterCrop(224),
        ]
    ),
    "Magnify 1.81": transforms.Compose(
        [
            PREPROCESS_TF,
            transforms.Resize(int(224 * 1.81), antialias=True),
            transforms.CenterCrop(224),
        ]
    ),
    "Magnify 1.92": transforms.Compose(
        [
            PREPROCESS_TF,
            transforms.Resize(int(224 * 1.92), antialias=True),
            transforms.CenterCrop(224),
        ]
    ),
    "Magnify 2.05": transforms.Compose(
        [
            PREPROCESS_TF,
            transforms.Resize(int(224 * 2.05), antialias=True),
            transforms.CenterCrop(224),
        ]
    ),
    "Magnify 2.25": transforms.Compose(
        [
            PREPROCESS_TF,
            transforms.Resize(int(224 * 2.25), antialias=True),
            transforms.CenterCrop(224),
        ]
    ),
    "Magnify 2.5": transforms.Compose(
        [
            PREPROCESS_TF,
            transforms.Resize(int(224 * 2.5), antialias=True),
            transforms.CenterCrop(224),
        ]
    ),
}
