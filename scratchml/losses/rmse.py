from numpy import mean, sqrt, zeros_like

from scratchml.losses.base import BaseLoss


class RMSE(BaseLoss):
    """Root Mean Squared Error Loss class."""
    def value(self, y_true, y_pred):
        """Calculate the value of the Root Mean Squared Error 
        loss function."""
        return sqrt(mean(self.error(y_true, y_pred) ** 2))

    def gradient(self, y_true, y_pred):
        """Calculate the gradient of the Root Mean Squared Error 
        loss function."""
        error = self.error(y_true, y_pred)
        rmse = self.value(y_true, y_pred)

        if rmse == 0:
            return zeros_like(error)
        
        return error / (error.size * rmse)