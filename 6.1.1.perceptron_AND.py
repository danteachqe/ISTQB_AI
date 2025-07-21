

import numpy as np
from sklearn.linear_model import Perceptron

# Define the AND function dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([0, 0, 0, 1])

# Create and train the Perceptron
clf = Perceptron(max_iter=10, tol=1e-3, random_state=42)
clf.fit(X, y)

# Evaluate the Perceptron
y_pred = clf.predict(X)
print("Predictions:", y_pred)
print("Actual:     ", y)
print("Weights:", clf.coef_)
print("Bias:", clf.intercept_)
print("Number of epochs:", clf.n_iter_)
print("Accuracy:", clf.score(X, y))


