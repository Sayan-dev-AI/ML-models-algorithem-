import numpy as np
from math import e



class Models:
    class SGD:
        m = 0
        b = 0 

        @classmethod
        def train(cls,x:np.ndarray, y:np.ndarray, epochs:int, learning_rate:float=0.001) -> np.ndarray:

            for _ in range(epochs):
                h_beta = cls.b + (cls.m*x)
                delta_0 = np.mean(h_beta - y)
                delta_1 = np.mean((h_beta - y) * x)
                
                cls.b -= learning_rate * delta_0
                cls.m -= learning_rate * delta_1

                mse = np.array(sum((y[i] - h_beta[i]) ** 2 for i in range(len(y))) / len(y))

                print(f'Epoch:{_+1} \tm: {cls.m}\tb: {cls.b}\tLoss: {mse:.20f}\n')

        def predict(self,x_predict:np.ndarray)->np.ndarray:

            output = self.b + (self.m * x_predict)
            
            return output
        

    
    class LogisticRegression:
        def __init__(self):
            self.m = 0
            self.b = 0

        def sigmoid(self, parameters: np.ndarray) -> np.ndarray:
            return 1 / (1 + np.exp(-parameters))

        def log_loss(self, y: np.ndarray, y_hat_pred: np.ndarray) -> float:
            n = y.size
            y_hat = self.sigmoid(y_hat_pred)
            loss = -(1/n) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
            return loss

        def train(self, x: np.ndarray, y: np.ndarray, epochs: int, learning_rate: float = 0.001) -> None:
            assert x.size == y.size, f'x{x.shape} is not equal to y{y.shape}'

            self.m = 0
            self.b = 0
            n = x.size

            for epoch in range(epochs):
                y_pred = self.sigmoid((self.m * x) + self.b)

                db = (1/n) * np.sum(y_pred - y)
                dm = (1/n) * np.sum((y_pred - y) * x)

                self.b -= learning_rate * db
                self.m -= learning_rate * dm

                loss_logistic = self.log_loss(y, y_pred)
                print(f'Epoch: {epoch + 1}\tm: {self.m}\tb: {self.b}\tLoss: {loss_logistic:.20f}\n')

        def predict(self, x_predict: np.ndarray) -> np.ndarray:
            pred = self.b + (self.m * x_predict)
            output = self.sigmoid(pred)
            return output
                    

    class OLS:
        m = 0
        b = 0

        @classmethod
        def train(cls,x: np.ndarray, y: np.ndarray) -> None:
            update_x = np.vstack([np.ones(x.size), x]).T
            beta = np.dot(np.linalg.pinv(update_x), y)

            cls.m = beta[1]
            cls.b = beta[0]

        def predict(self,x_predict:np.ndarray) -> np.ndarray:

            output = self.b + (self.m * x_predict)

            return output
    
    class least_squares_regression:
        m = 0
        b = 0

        @classmethod
        def train(cls,x:np.ndarray, y:np.ndarray ) -> None:

            x_ = (1/len(x))*x.sum()
            y_ = (1/len(y))*y.sum()

            cls.m = np.sum([(xi-x_)*(yi-y_)for xi,yi in zip(x,y)]) / np.sum([(xi-x_)**2 for xi in x])
            cls.b = y_ - (cls.m*x_)

        def predict(self, x_predict:np.ndarray) -> np.ndarray:

            output = self.b + (self.m*x_predict)
            return output

        

            















































