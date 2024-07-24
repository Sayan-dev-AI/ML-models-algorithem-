import numpy as np
import matplotlib.pyplot as plt

def ols(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    update_x = np.vstack([np.ones(x.size), x]).T
    beta = np.dot(np.linalg.pinv(update_x), y)
    return beta

# Generate synthetic data for testing
from sklearn.datasets import make_regression
x, y = make_regression(n_samples=100, n_features=1, noise=16, random_state=42)
x = x.flatten()

# Compute OLS regression coefficients
beta = ols(x, y)
b, m = beta[0], beta[1]

# Predict y values using the regression line
y_pred = (m * x) + b

# Plotting the data and regression line
plt.scatter(x, y, color='blue', label='Data points')
plt.plot(x, y_pred, color='green', label='Regression line')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

print(f"Intercept (b): {b}")
print(f"Slope (m): {m}")




























