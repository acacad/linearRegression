import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

# 1. get the data
data = pd.read_csv('abalone.data', header=None)
data.columns = ["Sex", "Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight", "Shell weight", "Rings"]

# not using "Sex" et "Height"
data = data.drop(["Sex", "Height"], axis=1)

# Convert data en numpy array
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# shuffle data
X, y = shuffle(X, y, random_state=0)

# Devide data in two equals parts
split_index = len(data) // 2
X_train, X_val = X[:split_index], X[split_index:]
y_train, y_val = y[:split_index], y[split_index:]

# Normalize features
X_train_mean = X_train.mean(axis=0)
X_train_std = X_train.std(axis=0)
X_train = (X_train - X_train_mean) / X_train_std
X_val = (X_val - X_train_mean) / X_train_std  # Utiliser les paramètres d'entraînement pour la validation

#
#add column for bayes
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_val = np.c_[np.ones(X_val.shape[0]), X_val]

# Initialise the weights
m_train, n = X_train.shape
theta = np.zeros(n)

# 3)implementing linear regression with gradient-descent
    #cost function:
def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    error = predictions - y
    cost = (1 / (2 * m)) * np.sum(error ** 2)
    return cost

#gradient descent function
def gradient_descent(X, y, theta, learning_rate, epochs):
    m = len(y)
    cost_history = np.zeros(epochs)

    for i in range(epochs):
        predictions = X.dot(theta)
        error = predictions - y
        theta -= (learning_rate / m) * (X.T.dot(error))
        cost_history[i] = compute_cost(X, y, theta)

    return theta, cost_history


def r_squared(X, y, theta):
    predictions = X.dot(theta)
    ss_total = np.sum((y - np.mean(y)) ** 2)
    ss_residual = np.sum((y - predictions) ** 2)
    return 1 - (ss_residual / ss_total)

# Paramètres for gradient descent
learning_rate = 0.01
epochs = 1000

# train the model
theta, cost_history = gradient_descent(X_train, y_train, theta, learning_rate, epochs)

# Predictions on  validation-data and training-data
predictions_train = X_train.dot(theta)
predictions_val = X_val.dot(theta)




# Calcul  % of mistakes on the prediction
error = np.abs(predictions_val - y_val)
percentage_correct = 100 - (np.mean(error) / np.mean(y_val) * 100)

# R-squared
r2_train = r_squared(X_train, y_train, theta)
r2_val = r_squared(X_val, y_val, theta)

# print results
print("Coefficients:", theta)

#print("Coût final:", cost_history[-1])
print(f"Pourcentage de prédictions correctes : {percentage_correct:.2f}%")
#print(f"Pourcentage de prédictions avec marge ±1 : {percentage_correct_with_margin:.2f}%")
print(f"R-SQUARED-TRAINING : {r2_train:.4f}")
print(f"R-SQUARED-VALIDATION : {r2_val:.4f}")

# cost function representation during the epochs
plt.plot(range(epochs), cost_history, 'b-')
plt.title('COST depending on the EPOCH')
plt.xlabel('EPOCH')
plt.ylabel('COST')
plt.grid(True)
plt.show()
