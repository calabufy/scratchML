from numpy import mean, sum

from scratchml.metrics.base import BaseMetric


class R2(BaseMetric):
    """R-squared class."""

    def error(self, y_true, y_pred):
        return y_pred - y_true

    def value(self, y_true, y_pred):
        """Calculate the value of the R-squared metric."""
        ss_res = sum(self.error(y_true, y_pred) ** 2)
        ss_tot = sum((y_true - mean(y_true)) ** 2)

        if ss_tot == 0:
            return 1.0

        return 1 - (ss_res / ss_tot)

    def score(self, y_true, y_pred):
        return self.value(y_true, y_pred)