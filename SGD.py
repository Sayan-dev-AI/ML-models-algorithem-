import numpy as np
import matplotlib.pyplot as plt

def gradient(x:np.ndarray, y:np.ndarray, epochs:int, learning_rate:float=0.001) -> np.ndarray:
    beta = np.array([0.0, 0.0])
    m = len(x)  # Number of data points

    for _ in range(epochs):
        h_beta = beta[0] + (beta[1]*x)
        delta_0 = np.mean(h_beta - y)
        delta_1 = np.mean((h_beta - y) * x)
        
        beta[0] -= learning_rate * delta_0
        beta[1] -= learning_rate * delta_1

        mse = np.array(sum((y[i] - h_beta[i]) ** 2 for i in range(len(y))) / len(y))

        # print(f'Epoch:{_+1} \tm: {beta[1]}\tb: {beta[0]}\tLoss: {mse:.20f}\n')


    return beta



from sklearn.datasets import make_regression
x, y = make_regression(n_samples=100, n_features=1, noise=16, random_state=42)
x = x.flatten()

# Compute OLS regression coefficients
beta = gradient(x, y, epochs=6000)
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






