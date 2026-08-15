class BaseLoss:
    def error(self, y_true, y_pred):
        """Calculate the error between the predicted and true values.
        Attention: errors must be calculated as y_pred - y_true, 
        because the gradient is calculated as dL/dy_pred, not dL/dy_true.
        
        Args:
            y_true (numpy.ndarray): The true values.
            y_pred (numpy.ndarray): The predicted values.

        Returns:
            numpy.ndarray: The error between the predicted and true values.
        """
        return y_pred - y_true

    def value(self, y_true, y_pred):
        """Calculate the value of the loss function.

        Args:
            y_true (numpy.ndarray): The true values.
            y_pred (numpy.ndarray): The predicted values.

        Returns:
            float: The value of the loss function.
        """
        raise NotImplementedError

    def gradient(self, y_true, y_pred):
        """Calculate the gradient of the loss function with respect to the
          predicted values.

        Args:
            y_true (numpy.ndarray): The true values.
            y_pred (numpy.ndarray): The predicted values.

        Returns:
            numpy.ndarray: The gradient of the loss function with respect 
            to the predicted values.
        """
        raise NotImplementedError