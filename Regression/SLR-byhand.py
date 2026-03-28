import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()

x = np.array(housing.data[:, 0])
y = np.array(housing.target)

# this is the dataset ^

def train(x,y):
    n = len(x)
    x_mean = sum(x)/n
    y_mean = sum(y)/n
# these are the mean values now we need to calculate the slope n intercept
# for that we need to calc the numerator and denominator 

    numerator =  np.sum((x-x_mean)*(y-y_mean)) # whats the covariance of x and y
    denominator = np.sum((x-x_mean)**2)   # how spread out x is / variance of x

    m = numerator/denominator
    b = y_mean - m * x_mean
    return m, b


def predict(x,m,b):
    y = m * x + b
    return y


def mse(y,y_pred):
    mse = np.sum((y-y_pred)**2) / len(y)
    return mse


# run it
m, b = train(x, y)
print(f"slope (m): {m:.2f}, intercept (b): {b:.2f}")

# first 5 predictions 
y_pred = predict(x, m, b)
print(f"predictions (first 5): {y_pred[:5]}")

error = mse(y, y_pred)
print(f"MSE: {error:.2f}")

# visualise the dataset

# scatter plot of actual data
plt.scatter(x, y, alpha=0.3, label='actual')
# regression line
plt.plot(x, y_pred, color='red', label='prediction')
plt.xlabel('Median Income')
plt.ylabel('House Price')
plt.legend()
plt.show()



