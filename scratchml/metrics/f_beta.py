from scratchml.metrics.base import BaseMetric
from scratchml.metrics.precision import Precision
from scratchml.metrics.recall import Recall
from numpy import asarray

class FBeta(BaseMetric):
    def __init__(self, beta=1.0):
        if beta <= 0:
            raise ValueError("Beta must be greater than 0.")
        self.beta = beta

    def value(self, y_true, y_pred):
        y_true = asarray(y_true).reshape(-1)
        y_pred = asarray(y_pred).reshape(-1)

        if y_true.shape != y_pred.shape:
            raise ValueError("Y_true and Y_pred must be the same shape.")
        
        precision = Precision().value(y_true, y_pred)
        recall = Recall().value(y_true, y_pred)

        if precision + recall == 0:
            return 0.0

        beta_squared = self.beta ** 2
        f_beta_score = (1 + beta_squared) * (precision * recall) / (beta_squared * precision + recall)
        return f_beta_score

    def score(self, y_true, y_pred):
        return self.value(y_true, y_pred)