from numpy import asarray, column_stack, ones, zeros, sign, abs, sum
from scratchml.models.base import BaseModel
from scratchml.losses import MSE


class LinearRegression(BaseModel):
    def __init__(self, learning_rate=0.001, epochs=1000, l1=0.0, l2=0.0):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l1 = l1
        self.l2 = l2
        self.coefficients_ = None
        self.history_ = []

    def fit(self, X, y, loss_function=None):
        """Fit the linear regression model to the training data."""
        # If no loss function is provided, use 
        # Mean Squared Error (MSE) as the default.
        if loss_function is None:
            loss_function = MSE()

        X = asarray(X, dtype=float)
        y = asarray(y, dtype=float).reshape(-1)

        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y must be the same.")

        # Augment the input features with a bias term (intercept)
        X_augmented = column_stack((ones(X.shape[0]), X))
        self.coefficients_ = zeros(X_augmented.shape[1])
        self.history_ = []

        # Perform gradient descent to optimize the coefficients
        for _ in range(self.epochs):
            # Compute predictions and gradients
            y_pred = X_augmented @ self.coefficients_
            # Compute the gradient of the loss function with respect to the coefficients
            loss_gradient = loss_function.gradient(y, y_pred)
            gradient = X_augmented.T @ loss_gradient

            # Add regularization gradients for L1 and L2 penalties
            regularization_gradient = zeros(self.coefficients_.shape)
            regularization_gradient[1:] += self.l1 * sign(self.coefficients_[1:])
            regularization_gradient[1:] += 2 * self.l2 * self.coefficients_[1:]

            # Update coefficients using the learning rate and the computed gradient
            gradient += regularization_gradient
            self.coefficients_ -= self.learning_rate * gradient

            # Compute and store the loss value for the current epoch
            y_pred = X_augmented @ self.coefficients_
            loss_value = loss_function.value(y, y_pred)
            # Add regularization loss to the total loss
            regularization_loss = self.l1 * sum(abs(self.coefficients_[1:]))
            regularization_loss += self.l2 * sum(self.coefficients_[1:] ** 2)
            self.history_.append(loss_value + regularization_loss)

        return self

    def predict(self, X):
        """Predict the target values for the given input features."""
        X = asarray(X, dtype=float)
        X_augmented = column_stack((ones(X.shape[0]), X))
        return X_augmented @ self.coefficients_

    def score(self, X, y, loss_function):
        """Evaluate the model's performance on the given data."""
        y_pred = self.predict(X)
        return loss_function.value(y, y_pred)
