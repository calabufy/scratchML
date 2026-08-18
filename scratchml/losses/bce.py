from scratchml.losses.base import BaseLoss
from numpy import exp, mean, logaddexp, empty_like

class BCE(BaseLoss):
    """Binary Cross-Entropy Loss class."""
    def sigmoid(self, logits):
        """Sigmoid function turns logits (continuous regression value)
          into probabilities. In this function we use a numerically 
          stable implementation of the sigmoid function."""
        probs = empty_like(logits, dtype=float)
        positive_mask = logits >= 0
        negative_mask = ~positive_mask

        probs[positive_mask] = 1 / (1 + exp(-logits[positive_mask]))
        probs[negative_mask] = exp(logits[negative_mask]) / (1 + exp(logits[negative_mask]))
        return probs

    def value(self, y_true, logits):
        prob = self.sigmoid(logits)
        return mean(logaddexp(0.0, logits) - y_true * logits)

    def gradient(self, y_true, logits):
        prob = self.sigmoid(logits)
        return (prob - y_true) / y_true.size