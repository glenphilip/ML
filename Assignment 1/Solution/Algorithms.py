import numpy as np
import matplotlib.pyplot as plt



"""Gradient Descent class"""

class gradient_descent:
    def __init__(self, X_train, y_train, w_ml, epochs):
        self.X = X_train
        self.y = y_train
        self.w_ml = w_ml
        self.epochs = epochs
        self.w = np.zeros(self.X.shape[1])

    def calculate_weights(self):
        errors = []
        for t in range(1, self.epochs+1):
            step_size = 1/t
            gradient = (1 / len(self.X)) * (self.X.T @ (self.X @ self.w - self.y))
            self.w = self.w - step_size * gradient
            e = np.linalg.norm(self.w - self.w_ml)
            errors.append(e)

        return self.w, errors

    def mse(self):
      y_pred = self.X @ self.w
      return np.mean((self.y - y_pred)**2)



"""Stochastic Gradient Descent class"""

class stochastic_gradient_descent:
    def __init__(self, X_train, y_train, w_ml, epochs, batch_size=100):
        self.X = X_train
        self.y = y_train
        self.w_ml = w_ml
        self.epochs = epochs
        self.batch_size = batch_size
        self.w = np.zeros(self.X.shape[1])

    def calculate_weights(self):
        errors = []
        np.random.seed(25)
        for t in range(1, self.epochs + 1):
            indices = np.arange(len(self.y))
            np.random.shuffle(indices)
            indices = list(indices)

            while indices:
                batch = indices[:self.batch_size]
                indices = indices[self.batch_size:]
                X_batch = self.X[batch]
                y_batch = self.y[batch]
                step_size = 1 /t
                gradient = (1 / len(X_batch)) * (X_batch.T @ (X_batch @ self.w - y_batch))
                self.w = self.w - step_size * gradient
                e = np.linalg.norm(self.w - self.w_ml)
                errors.append(e)

        return self.w, errors

    def mse(self):
        y_pred = self.X @ self.w
        return np.mean((self.y - y_pred) ** 2)



"""Function to split dataset into train and validation sets for cross-validation"""

def split_dataset(X, y, fold, num_folds):
    sz = int(len(X) / num_folds)
    indices = np.arange(len(X))
    val = indices[fold * sz:(fold + 1) * sz]
    X_val, y_val = X[val], y[val]
    train = [i for i in indices if i not in val]
    X_train, y_train = X[train], y[train]

    return X_train, y_train, X_val, y_val



"""Ridge Regression class"""

class ridge_regression:
    def __init__(self, X_train, y_train, X_test, y_test, epochs=1000, folds=5):
        self.X = X_train
        self.y = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.epochs = epochs
        self.folds = folds

    def ridge(self, X, y, lambda_value):
        w = np.zeros(self.X.shape[1])
        for t in range(1, self.epochs + 1):
            step_size = 1/t
            gradient = (2/len(self.X))*((X.T @ (X @ w - y)) + (lambda_value * w))
            w = w - step_size * gradient
        return w

    def cv(self, lambdas):
        cv_errors = []
        for l in lambdas:
            errors = []
            for f in range(self.folds):
                X_train, y_train, X_val, y_val = split_dataset(self.X, self.y, f, self.folds)
                w = self.ridge(X_train, y_train, l)
                y_pred = X_val @ w
                errors.append(np.mean((y_val - y_pred) ** 2))

            cv_errors.append(np.mean(errors))

        return cv_errors

    def calculate_weights(self):
        lambdas1 = np.arange(0,51)
        errors1 = self.cv(lambdas1)
        lambda1 = lambdas1[errors1.index(min(errors1))]
        lambdas2 = np.linspace(lambda1 - 2, lambda1 + 2, 81)
        errors2 = self.cv(lambdas2)
        lambda2 = lambdas2[errors2.index(min(errors2))]
        w_R = self.ridge(self.X, self.y, lambda2)
        return w_R, errors1, lambda2



"""Kernel regression class"""

class kernel_regression:
    def __init__(self, X_train, y_train, X_test, y_test, num_folds=5):
        self.X = X_train
        self.y = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.num_folds = num_folds

    def cv(self):
        degrees = np.arange(11)
        best_d = None
        mse = []

        for d in degrees:
            error = []
            for fold in range(self.num_folds):
                X_train, y_train, X_val, y_val = split_dataset(self.X, self.y, fold, self.num_folds)
                k = ((X_train @ X_train.T) + 1) ** d
                A = np.linalg.pinv(k) @ y_train
                y_pred = self.predict(X_train, X_val, A, d)
                error.append(np.mean((y_val - y_pred) ** 2))

            mse.append(np.mean(error))

            if np.mean(error) <= min(mse):
                best_d = d

        print("Best polynomial kernel is of degree:", best_d, "with MSE:",np.round(np.min(error),4))
        return best_d, mse
    
    def predict(self, X, X_test, A, d):
        K = ((X @ X_test.T) + 1) ** d
        return K.T @ A

    def alpha(self):
        best_d, errors = self.cv()
        k = ((self.X @ self.X.T) + 1) ** best_d
        A = np.linalg.pinv(k) @ self.y
        return best_d, A, errors
