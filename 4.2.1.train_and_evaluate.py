import pandas as pd
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

print("Generating a complex, unbalanced dataset...")
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=6,
    n_redundant=2,
    n_classes=3,
    n_clusters_per_class=1,
    weights=[0.7, 0.2, 0.1],
    flip_y=0.01,
    random_state=42
)
print("Dataset generated.")

print("Scaling features...")
scaler = StandardScaler()
X = scaler.fit_transform(X)
print(f"Features scaled. Shape: {X.shape}")

print("Splitting data into train (60%), validation (20%), and test (20%) sets...")
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)
print(f"Train set size: {X_train.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

print("Training model...")
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
print("Training complete.")

model_dir = os.path.join(os.path.dirname(__file__), "complex_logreg_model.joblib")
joblib.dump(model, model_dir)
print(f"Model saved to {model_dir}")

print("Evaluating on validation set...")
y_val_pred = model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_accuracy:.4f}")

print("Evaluating on test set...")
y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")

print("\nExplanation:")
print("This dataset is more complex and unbalanced, so you may see different accuracy values for validation and test sets. Validation accuracy is used for tuning, while test accuracy shows final generalization.")
