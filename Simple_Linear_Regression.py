import numpy as np
import matplotlib.pyplot as plt

def mean(x:np.ndarray,y:np.ndarray) -> np.ndarray:
    
    assert x.size == y.size, f'x{x.shape} is not equeals to y{y.shape}'

    x_ = np.array((1/x.size)*x.sum())
    y_ = np.array((1/y.size)*y.sum())

    return x_, y_
def slope(x:np.array,y:np.ndarray) -> np.ndarray:

    assert x.size == y.size, f'x{x.shape} is not equeals to y{y.shape}'

    x_, y_ = mean(x,y)
    
    neumerator = np.sum([(xi-x_)*(yi-y_)for xi,yi in zip(x,y)])
    denomenator = np.sum([(xi-x_)**2 for xi in x])

    slope = neumerator/denomenator

    return slope

def intercept(x:np.array,y:np.ndarray) -> np.ndarray:

    assert x.size == y.size, f'x{x.shape} is not equeals to y{y.shape}'

    x_, y_ = mean(x,y)
    m = slope(x,y)

    interception = y_- (m*x_)

    return interception

def Simple_linear_Regression(x:np.array,y:np.ndarray) -> np.ndarray:

    assert x.size == y.size, f'x{x.shape} is not equeals to y{y.shape}'

    m_slope = slope(x,y)
    b_intercept = intercept(x,y)

    return b_intercept,m_slope



from sklearn.datasets import make_regression
x, y = make_regression(n_samples=100, n_features=1, noise=16, random_state=42)
x = x.flatten()

# Compute OLS regression coefficients
beta = Simple_linear_Regression(x, y)
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


