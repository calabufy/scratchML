from numpy import asarray, unique
from scratchml.metrics.base import BaseMetric

class Gini(BaseMetric):
    """Gini coefficient metric class."""
    def value(self, y):
        y = asarray(y).reshape(-1)
        _, counts = unique(y, return_counts=True)
        probabilities = counts / y.size
        return 1.0 - sum(probabilities ** 2)

    def score(self, y):
        return self.value(y)