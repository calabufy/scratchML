from numpy import asarray, concatenate, exp
from numpy.random import default_rng
from scratchml.models import LogisticRegression
from scratchml.losses import BCE
from scratchml.metrics import Precision, Recall, F1

rng = default_rng(seed=42)

# Generate synthetic data for logistic regression
true_bias = -0.25
true_weights = asarray([1.5, -2.0])

def generate_synthetic_data(n_samples=100):
    # Generate X points uniformly in the range [-5, 5] for two features
    X = rng.uniform(-5.0, 5.0, size=(n_samples, 2))
    # Compute the logits using the true coefficients and bias
    logits = true_bias + X @ true_weights
    # Convert logits to probabilities using the sigmoid function
    probabilities = BCE().sigmoid(logits)
    # Generate binary target values based on the probabilities
    y = rng.binomial(1, probabilities)
    return X, y

# Generate training and test datasets
X, y = generate_synthetic_data(n_samples=1000)
X_test, y_test = generate_synthetic_data(n_samples=200)

# Create and train the logistic regression model
model = LogisticRegression(learning_rate=0.05, epochs=1000)
model.fit(X, y, loss_function=BCE())

# Evaluate the model on the training data
print("True Coefficients:", concatenate(([true_bias], true_weights)))
print("Model Coefficients:", model.coefficients_)
print("\nTraining Metrics:")
print("BCE Loss:", round(BCE().value(y, model.predict_proba(X)), 4))
print("Precision:", round(Precision().value(y, model.predict(X)), 4))
print("Recall:", round(Recall().value(y, model.predict(X)), 4))
print("F1_Score:", round(F1().score(y, model.predict(X)), 4))

# Evaluate the model on the test data
print("\nTest Metrics:")
print("BCE Loss:", round(BCE().value(y_test, model.predict_proba(X_test)), 4))
print("Precision:", round(Precision().value(y_test, model.predict(X_test)), 4))
print("Recall:", round(Recall().value(y_test, model.predict(X_test)), 4))
print("F1_Score:", round(F1().score(y_test, model.predict(X_test)), 4))