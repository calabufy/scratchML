from scratchml.models.base import BaseModel
from scratchml.losses import BCE
from numpy import asarray, column_stack, ones, zeros, sign, mean

class LogisticRegression(BaseModel):
    def __init__(self, learning_rate=0.001, epochs=1000, l1=0.0, l2=0.0):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l1 = l1
        self.l2 = l2
        self.coefficients_ = None
        self.history_ = []

    def fit(self, X, y, loss_function=None):
        if loss_function is None:
            loss_function = BCE()

        X = asarray(X, dtype=float)
        y = asarray(y, dtype=float).reshape(-1)

        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y must be the same.")

        if not all((y == 0) | (y == 1)):
            raise ValueError("Target values must be binary (0 or 1).")

        X_augmented = column_stack((ones(X.shape[0]), X))
        self.coefficients_ = zeros(X_augmented.shape[1])
        self.history_ = []

        for _ in range(self.epochs):
            logits = X_augmented @ self.coefficients_

            loss_gradient = loss_function.gradient(y, logits)
            gradient = X_augmented.T @ loss_gradient

            regularization_gradient = zeros(self.coefficients_.shape)
            regularization_gradient[1:] += self.l1 * sign(self.coefficients_[1:])
            regularization_gradient[1:] += 2 * self.l2 * self.coefficients_[1:]

            gradient += regularization_gradient
            self.coefficients_ -= self.learning_rate * gradient

            logits = X_augmented @ self.coefficients_
            loss_value = loss_function.value(y, logits)
            regularization_loss = self.l1 * sum(abs(self.coefficients_[1:]))
            regularization_loss += self.l2 * sum(self.coefficients_[1:] ** 2)
            self.history_.append(loss_value + regularization_loss)

        return self

    def predict_proba(self, X):
        X = asarray(X, dtype=float)
        X_augmented = column_stack((ones(X.shape[0]), X))
        logits = X_augmented @ self.coefficients_
        probabilities = BCE().sigmoid(logits)
        return probabilities

    def predict(self, X, threshold=0.5):
        if not (0.0 <= threshold <= 1.0):
            return ValueError("Trashold must be in [0.0, 1.0].")
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

    def score(self, X, y, loss_function=None):
        y_pred = self.predict(X)
        if loss_function is None:
            return mean(y_pred == y)
        return loss_function.value(y, y_pred)