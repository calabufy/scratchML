from scratchml.metrics.base import BaseMetric
from numpy import asarray, sum

class Recall(BaseMetric):
    def value(self, y_true, y_pred):
        y_true = asarray(y_true).reshape(-1)
        y_pred = asarray(y_pred).reshape(-1)

        if y_true.shape != y_pred.shape:
            raise ValueError("Y_true and Y_pred must be same shape.")

        TP = sum((y_true == 1) & (y_pred == 1))
        FN = sum((y_true == 1) & (y_pred == 0))

        return TP / (TP + FN) if TP + FN != 0 else 0.0

    def score(self, y_true, y_pred):
        return self.value(y_true, y_pred)