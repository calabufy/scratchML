from numpy import asarray, concatenate
from numpy.random import default_rng

from scratchml.models import LinearRegression
from scratchml.losses import RMSE, MAE
from scratchml.metrics import R2

# Generate synthetic data for linear regression
rng = default_rng(seed=42)
X = rng.uniform(0.0, 10.0, size=(100, 1))
true_bias = 2.0
true_weights = asarray([3.0])
noise = rng.normal(loc=0.0, scale=0.5, size=X.shape[0])
y = true_bias + X @ true_weights + noise

# Create and train the linear regression model
model = LinearRegression(learning_rate=0.01, epochs=1000, l1=0.1, l2=0.1)
model.fit(X, y, loss_function=RMSE())

# Evaluate the model on the training data
print("Coefficients:", model.coefficients_)
print("True Coefficients:", concatenate(([true_bias], true_weights)))
print("\nTraining Metrics:")
print("RMSE Loss:", round(RMSE().value(y, model.predict(X)), 4))
print("MAE Loss:", round(MAE().value(y, model.predict(X)), 4))
print("R2 Score:", round(R2().value(y, model.predict(X)), 4))

# Generate synthetic test data
X_test = rng.uniform(0.0, 10.0, size=(50, 1))
test_noise = rng.normal(loc=0.0, scale=0.5, size=X_test.shape[0])
y_test = true_bias + X_test @ true_weights + test_noise

# Evaluate the model on the test data
print("\nTest Metrics:")
print("RMSE Loss:", round(RMSE().value(y_test, model.predict(X_test)), 4))
print("MAE Loss:", round(MAE().value(y_test, model.predict(X_test)), 4))
print("R2 Score:", round(R2().value(y_test, model.predict(X_test)), 4))
