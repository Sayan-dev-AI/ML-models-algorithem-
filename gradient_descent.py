import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(x:np.ndarray, y:np.ndarray,epochs:int, learning_rate:float = 0.001) -> np.ndarray:
    
    assert x.size == y.size, f'x{x.shape} is not equeals to y{y.shape}'

    m = 0
    b = 0
    n = x.size
    

    for _ in range(epochs):

        y_pred = (m*x) + b

        db = (1/n) * np.sum(y_pred-y)
        dm = (1/n) * np.sum((y_pred-y)*x)

        b -= learning_rate * db
        m -= learning_rate * dm

        mse = np.array(sum((y[i] - y_pred[i]) ** 2 for i in range(len(y))) / len(y))

        # print(f'Epoch:{_+1} \tm: {m}\tb: {b}\tLoss: {mse:.20f}\n')


    return np.array([b,m])

# x = np.array([1,3,7,9,11,19,35,47,67,90])
# y = (0.3*x) + 0.2

# b,m = gradient_descent(x,y,epochs=59135, learning_rate=0.001)

# x_ = np.arange(0,100,2)
# y_ = (m*x_) + b

from sklearn.datasets import make_regression
x, y = make_regression(n_samples=100, n_features=1, noise=16, random_state=42)
x = x.flatten()

# Compute OLS regression coefficients
beta = gradient_descent(x, y, epochs=6000)
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









