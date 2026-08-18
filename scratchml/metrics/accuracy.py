from numpy import asarray, mean
from scratchml.metrics.base import BaseMetric


class Accuracy(BaseMetric):
    """Accuracy metric class."""
    def value(self, y_true, y_pred):
        y_true = asarray(y_true).reshape(-1)
        y_pred = asarray(y_pred).reshape(-1)

        if y_true.shape != y_pred.shape:
            raise ValueError("Y_true and Y_pred must be the same shape.")

        return float(mean(y_true == y_pred))

    def score(self, y_true, y_pred):
        return self.value(y_true, y_pred)
