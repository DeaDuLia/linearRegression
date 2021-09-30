# Making imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12.0, 9.0)
# Preprocessing Input data
data = pd.read_csv('Chicago_hotels.csv')
X = data.iloc[:, 2]
Y = data.iloc[:, 6]
# Building the model
X_mean = np.mean(X)
Y_mean = np.mean(Y)
num = 0
den = 0
for i in range(len(X)):
    num += (X[i] - X_mean)*(Y[i] - Y_mean)
    den += (X[i] - X_mean)**2
m = num / den
c = Y_mean - m*X_mean
Y_pred = m*X + c

plt.scatter(X, Y) # actual
plt.plot(X, Y_pred, color='red')
plt.show()

x_pred = [113, 114, 115, 116, 117, 118, 119, 120]
y_pred = [i * m + c for i in x_pred]
plt.plot(x_pred, y_pred, color='red')
plt.show()
print(y_pred)








