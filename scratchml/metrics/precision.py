from scratchml.metrics.base import BaseMetric
from numpy import asarray, sum

class Precision(BaseMetric):
    def value(self, y_true, y_pred):
        y_true = asarray(y_true).reshape(-1)
        y_pred = asarray(y_pred).reshape(-1)

        if y_true.shape != y_pred.shape:
            raise ValueError("Y_true and Y_pred must be the same shape.")

        TP = sum((y_pred == 1) & (y_true == 1))
        FP = sum((y_pred == 1) & (y_true == 0))

        return TP / (TP + FP) if TP + FP != 0 else 0.0

    def score(self, y_true, y_pred):
        return self.value(y_true, y_pred)