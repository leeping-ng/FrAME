from scipy import stats
import torch
from typing import Dict, List, Tuple


class KSTest:
    def __init__(self, num_classes: int, alpha: float = 0.05) -> None:
        """
        Multiple univariate Kolmogorov-Smirnov test.

        Args:
            num_classes (int): Number of classes for image classification
            alpha (float): Significance level to accept/reject null hypothesis
        """
        self.num_classes = num_classes
        self.alpha = alpha

    def perform_test(
        self, source_softmax: torch.Tensor, target_softmax: torch.Tensor
    ) -> Tuple[float, float, bool]:
        """
        Performs the K-S test on a source and target softmax.

        Args:
            source_softmax (torch.Tensor): Softmax tensor from source distribution
            target_softmax (torch.Tensor): Softmax tensor from target distribution

        Returns:
            ks_signal (float): Signal from K-S test
            ks_pvalue (float): P-value from K-S test
            shift_detected (bool): Whether a shift was detected
        """
        (
            source_softmax_by_class,
            target_softmax_by_class,
        ) = self._extract_softmaxes_by_class(source_softmax, target_softmax)

        ks_result = {}
        shift_detected = False
        ks_signal = 0
        ks_pvalue = 0
        for i in range(self.num_classes):
            ks_result[i] = stats.ks_2samp(
                source_softmax_by_class[i], target_softmax_by_class[i]
            )
            # Reject null hypothesis if any p-value < Bonferroni corrected significance level
            if ks_result[i].pvalue < self.alpha / self.num_classes:
                shift_detected = True

            ks_signal += ks_result[i].statistic
            ks_pvalue += ks_result[i].pvalue
        ks_signal /= self.num_classes
        ks_pvalue /= self.num_classes

        return ks_signal, ks_pvalue, shift_detected

    def _extract_softmaxes_by_class(
        self, source_softmax: torch.Tensor, target_softmax: torch.Tensor
    ) -> Tuple[Dict[int, List[float]], Dict[int, List[float]]]:
        """
        Extracts elements from softmax vector into a dictionary of softmax lists grouped by
        class, required for multiple K-S tests. Also handles batching of data.

        Args:
            source_softmax (torch.Tensor): Softmax tensor from source distribution
            target_softmax (torch.Tensor): Softmax tensor from target distribution

        Returns:
            source_softmax_by_class (Dict[int, List[float]]): Softmax from source distribution
                grouped by class
            target_softmax_by_class (Dict[int, List[float]]): Softmax from target distribution
                grouped by class
        """
        source_softmax_by_class = {}
        target_softmax_by_class = {}

        for i in range(self.num_classes):
            source_softmax_by_class[i] = []
            target_softmax_by_class[i] = []

        # loop over batches to aggregate softmax
        for i, (source_batch, target_batch) in enumerate(
            zip(source_softmax, target_softmax)
        ):
            source_batch = source_batch.numpy()
            target_batch = target_batch.numpy()

            for j in range(self.num_classes):
                source_softmax_by_class[j].extend(list(source_batch[:, j].squeeze()))
                target_softmax_by_class[j].extend(list(target_batch[:, j].squeeze()))

        return source_softmax_by_class, target_softmax_by_class
