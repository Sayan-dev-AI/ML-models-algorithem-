import numpy as np
from math import e
import matplotlib.pyplot as plt


def sigmoid(parameters:np.ndarray)-> np.ndarray:

      z = 1/(1+e**-(parameters))

      return z



def log_loss(y:np.ndarray,y_hat_pred:np.ndarray) -> np.ndarray:

    n = x.size

    y_hat = sigmoid(y_hat_pred)

    # loss = -(1/n)*sum((y*np.log(y_hat))+((1-y)*np.log(1-y_hat)))
    loss = -(1/n) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

    return loss


class logistic_gradient_descent:
    def __init__(self):
    
        pass

    def train(self,x:np.ndarray, y:np.ndarray,epochs:int, learning_rate:float = 0.001) -> np.ndarray:

        assert x.size == y.size, f'x{x.shape} is not equeals to y{y.shape}'

        self.m = 0
        self.b = 0
        n = x.size
        

        for _ in range(epochs):

            y_pred = sigmoid((self.m*x) + self.b)

            db = (1/n) * np.sum(y_pred-y)
            dm = (1/n) * np.sum((y_pred-y)*x)

            self.b -= learning_rate * db
            self.m -= learning_rate * dm



            # mse = np.array(sum((y[i] - y_pred[i]) ** 2 for i in range(len(y))) / len(y))
            loss_logistic = log_loss(y,y_pred)

            print(f'Epoch:{_+1} \tm: {self.m}\tb: {self.b}\tLoss: {loss_logistic:.20f}\n')


        return 0
    
    def predict(self,x_predict:np.ndarray)->np.ndarray:

        output =  sigmoid(self.b + (self.m*x_predict))
        return output


x = np.arange(0,stop=100, step=2)
y0 = np.zeros(25)
y1 = np.ones(25)

y = np.hstack([y0,y1])

model = logistic_gradient_descent()
model.train(x,y,epochs=120000)

y_pred = model.predict(x)

plt.scatter(x, y, color='blue', label='Data points')
plt.plot(x, y_pred, color='green', label='Regression line')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()


