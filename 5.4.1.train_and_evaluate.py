import pandas as pd
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import matplotlib.pyplot as plt
import numpy as np

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
print("Validation Classification Report:")
print(classification_report(y_val, y_val_pred, digits=4))

# --- Graphical report for validation set ---
val_report = classification_report(y_val, y_val_pred, output_dict=True)
labels = [str(i) for i in sorted(np.unique(y))]
precision = [val_report[label]['precision'] for label in labels]
recall = [val_report[label]['recall'] for label in labels]
f1 = [val_report[label]['f1-score'] for label in labels]
acc = [val_accuracy] * len(labels)

x = np.arange(len(labels))
width = 0.2
fig, ax = plt.subplots()
rects1 = ax.bar(x - 1.5*width, acc, width, label='Accuracy')
rects2 = ax.bar(x - 0.5*width, precision, width, label='Precision')
rects3 = ax.bar(x + 0.5*width, recall, width, label='Recall')
rects4 = ax.bar(x + 1.5*width, f1, width, label='F1-score')

ax.set_ylabel('Score')
ax.set_title('Validation Metrics by Class')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.ylim(0, 1.05)
plt.tight_layout()


plt.show()



# --- Evaluate on test set and plot metrics ---
print("Evaluating on test set...")
y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Test Accuracy: {test_accuracy:.4f}")
print("Test Classification Report:")
print(classification_report(y_test, y_test_pred, digits=4))

# --- Graphical report for test set ---
test_report = classification_report(y_test, y_test_pred, output_dict=True)
precision = [test_report[label]['precision'] for label in labels]
recall = [test_report[label]['recall'] for label in labels]
f1 = [test_report[label]['f1-score'] for label in labels]
acc = [test_accuracy] * len(labels)

fig, ax = plt.subplots()
rects1 = ax.bar(x - 1.5*width, acc, width, label='Accuracy')
rects2 = ax.bar(x - 0.5*width, precision, width, label='Precision')
rects3 = ax.bar(x + 0.5*width, recall, width, label='Recall')
rects4 = ax.bar(x + 1.5*width, f1, width, label='F1-score')

ax.set_ylabel('Score')
ax.set_title('Test Metrics by Class')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.ylim(0, 1.05)
plt.tight_layout()
plt.show()

# --- Final graph: average metrics across all classes ---
val_avg_precision = np.mean([val_report[label]['precision'] for label in labels])
val_avg_recall = np.mean([val_report[label]['recall'] for label in labels])
val_avg_f1 = np.mean([val_report[label]['f1-score'] for label in labels])
val_avg_acc = val_accuracy
test_avg_precision = np.mean([test_report[label]['precision'] for label in labels])
test_avg_recall = np.mean([test_report[label]['recall'] for label in labels])
test_avg_f1 = np.mean([test_report[label]['f1-score'] for label in labels])
test_avg_acc = test_accuracy

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
val_means = [val_avg_acc, val_avg_precision, val_avg_recall, val_avg_f1]
test_means = [test_avg_acc, test_avg_precision, test_avg_recall, test_avg_f1]

x = np.arange(len(metrics))
width = 0.35
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, val_means, width, label='Validation')
rects2 = ax.bar(x + width/2, test_means, width, label='Test')

ax.set_ylabel('Score')
ax.set_title('Average Metrics Across All Classes')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
plt.ylim(0, 1.05)
plt.tight_layout()
plt.show()

print("\nExplanation:")
print("This dataset is more complex and unbalanced, so you may see different accuracy values for validation and test sets. Validation accuracy is used for tuning, while test accuracy shows final generalization.")
