from scipy import stats


class KSTest:
    def __init__(self, num_classes, alpha=0.05):
        self.num_classes = num_classes
        self.alpha = alpha

    def _extract_softmaxes_by_class(self, source_softmax, target_softmax):
        # initialise empty data structures to be extended over batches
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

    def perform_test(self, source_softmax, target_softmax):
        (
            source_softmax_by_class,
            target_softmax_by_class,
        ) = self._extract_softmaxes_by_class(source_softmax, target_softmax)

        # K-S taking in softmax
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
