# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

# Step 1: Generate synthetic sine wave data with noise
X = np.linspace(-3, 3, 100)                             # 100 points evenly spaced between -3 and 3
y = np.sin(X) + np.random.normal(0, 0.1, 100)           # y = sin(x) + noise
X = X.reshape(-1, 1)                                    # Reshape for sklearn compatibility

# Step 2: Underfitting model - Linear Regression (too simple)
model_linear = LinearRegression()
model_linear.fit(X, y)                                  # Train on full data
y_pred_linear = model_linear.predict(X)                 # Predict on full X

# Step 3: Appropriate fit - Polynomial Regression (degree 5)
model_poly = make_pipeline(PolynomialFeatures(degree=5), LinearRegression())
model_poly.fit(X, y)                                    # Train with polynomial features
y_pred_poly = model_poly.predict(X)                     # Predict

# Step 4: Overfitting model - Decision Tree trained on tiny subset (7% of data)
X_train_over, _, y_train_over, _ = train_test_split(X, y, train_size=0.07, random_state=42)
model_tree = DecisionTreeRegressor(max_depth=10)        # High complexity
model_tree.fit(X_train_over, y_train_over)              # Train on very few points
y_pred_tree = model_tree.predict(X)                     # Predict on all data

# Step 5: Plot the original data and all model predictions
plt.figure(figsize=(10,6))
plt.scatter(X, y, color='blue', label='Actual Data', alpha=0.6)            # Original noisy data
plt.plot(X, y_pred_linear, color='orange', label='Underfit: Linear', linewidth=2)
plt.plot(X, y_pred_poly, color='green', label='Good Fit: Poly Degree 5', linewidth=2)
plt.plot(X, y_pred_tree, color='red', linestyle='--', label='Overfit: Decision Tree', linewidth=2)
plt.scatter(X_train_over, y_train_over, color='black', label='Overfit Training Points', zorder=10)

# Add title and legend
plt.title('Model Fit Comparison: Underfitting, Good Fit, Overfitting')
plt.legend()
plt.tight_layout()
plt.show()
