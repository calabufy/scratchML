from numpy import mean

from scratchml.losses.base import BaseLoss


class MSE(BaseLoss):
    """Mean Squared Error Loss class."""
    def value(self, y_true, y_pred):
        """Calculate the value of the Mean Squared Error
        loss function."""
        return mean(self.error(y_true, y_pred) ** 2)

    def gradient(self, y_true, y_pred):
        """Calculate the gradient of the Mean Squared Error
        loss function."""
        error = self.error(y_true, y_pred)
        return 2 * error / error.size