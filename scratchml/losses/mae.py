from numpy import mean, abs, sign

from scratchml.losses.base import BaseLoss


class MAE(BaseLoss):
    """Mean Absolute Error Loss class."""
    def value(self, y_true, y_pred):
        """Calculate the value of the Mean Absolute Error 
        loss function."""
        return mean(abs(self.error(y_true, y_pred)))
    
    def gradient(self, y_true, y_pred):
        """Calculate the gradient of the Mean Absolute Error
          loss function."""
        error = self.error(y_true, y_pred)
        return sign(error) / error.size