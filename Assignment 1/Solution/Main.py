import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Algorithms  import *     # file containing the classes of different regression algorithms



train_path = r"C:\Users\Admin\Desktop\test1\FMLA1Q1Data_train.csv"
test_path = r"C:\Users\Admin\Desktop\test1\FMLA1Q1Data_test.csv"



"""Loading the training dataset"""

train_data = pd.read_csv(train_path, header = None)
train_data.columns = ['x1', 'x2', 'y']
train_data['bias'] = 1    
X_train = train_data[['bias', 'x1', 'x2']].values
y_train = train_data['y'].values

w_ml = (np.linalg.inv((X_train.T) @ X_train)) @ ((X_train.T) @ (y_train))   # calculating least squares solution by analytical method

y_pred = X_train @ w_ml                       # predicting the training values using w_ml
mse_ml = np.mean((y_train - y_pred)**2)       # calculating MSE

print("Least squares solution, w_ml:", np.round(w_ml,3))
print("Training error (MSE) using w_ml:",np.round(mse_ml,3))



"""Gradient descent"""

gd = gradient_descent(X_train, y_train, w_ml, epochs = 500)   
w_gd, werrors_list = gd.calculate_weights()
mse_gd = gd.mse()

print("Solution by gradient descent algorithm, w_gd:", np.round(w_gd,3))
print("Training Error (MSE) using w_gd:",np.round(mse_gd,3) )

plt.plot(werrors_list)
plt.xlabel('Number of iterations (t)')
plt.ylabel('||W_t - W_ML||2')
plt.title('Gradient Descent Algorithm')
plt.show()



"""Stochastic Gradient Descent"""

sgd = stochastic_gradient_descent(X_train, y_train,w_ml, epochs=100, batch_size=100)
w_sgd, errors = sgd.calculate_weights()
mse_sgd = sgd.mse()

print("Solution by stochastic gradient descent algorithm, w_sgd:", np.round(w_sgd,3))
print("Training Error (MSE) using w_sgd:",np.round(mse_sgd,3) )

plt.plot(errors)
plt.xlabel('Number of iterations (t)')
plt.ylabel('||W_t - W_ML||2')
plt.title('Stochastic Gradient Descent Algorithm')
plt.show()



"""Loading the test dataset"""

test_data = pd.read_csv(test_path, header = None)
test_data.columns = ['x1', 'x2', 'y']
test_data['bias'] = 1
X_test = test_data[['bias', 'x1', 'x2']].values
y_test = test_data['y'].values



"""Ridge regression"""

ridge = ridge_regression(X_train, y_train, X_test, y_test, epochs=1000)
w_R, errors, l = ridge.calculate_weights()

print("Best value of lambda:", l)
print("Solution obtained by Ridge regression, W_R:", np.round(w_R,3))

y_pred = X_test @ w_R          # predicting using the ridge solution
print("Test error (MSE) using W_R", np.round(np.mean((y_pred - y_test)**2),3))

y_pred = X_test @ w_ml          # predicting using the analytical solution
print("Test error (MSE) using w_ml:",np.round(np.mean((y_test - y_pred)**2),3))

plt.plot(np.arange(51), errors)
plt.xticks(np.arange(0,51,5))
plt.xlabel('Lambda values')
plt.ylabel('Validation Error')
plt.grid(True)
plt.title('Cross Validation Errors for different values of Lambda')
plt.show()



"""Visualisation of y using features x1 and x2"""

plt.scatter(train_data['x1'], train_data['y'])
plt.xlabel('x1')
plt.ylabel('y')
plt.title('Plot of y vs x1')
plt.show()

plt.scatter(train_data['x2'], train_data['y'])
plt.xlabel('x2')
plt.ylabel('y')
plt.title('Plot of y vs x2')
plt.show()



"""Kernel regression"""

kernel = kernel_regression(X_train, y_train, X_test, y_test)
d, A, error = kernel.alpha()
y_pred = kernel.predict(X_train, X_test, A, d)   # prediction using polynomial kernel
e = np.mean((y_test - y_pred) ** 2)

print("Test Error(MSE) using polynomial kernel of degree",d,"is",np.round(e,4))

plt.plot(np.arange(11), error, "o-")
plt.xlabel("Degree")
plt.ylabel('Mean Squared Error')
plt.xticks(np.arange(11))
plt.title('MSE vs Degree of Polynomial Kernel')
plt.grid(True)
plt.show()