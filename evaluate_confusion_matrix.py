"""
This script demonstrates how to use the `evaluate` library from Hugging Face to calculate a 2x2 confusion matrix for binary classification results.

- The `evaluate` library provides a simple interface for common ML metrics.
- We simulate predictions and true labels for a binary classification task.
- The script prints the confusion matrix and explains its meaning.
"""

import evaluate
import numpy as np

import matplotlib.pyplot as plt

# Simulated binary classification results
# 1 = positive class, 0 = negative class
true_labels = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
predictions = [1, 0, 1, 0, 0, 1, 1, 0, 0, 0]

# Load the confusion matrix metric from Hugging Face evaluate
confusion_matrix_metric = evaluate.load("confusion_matrix")

# Compute the confusion matrix
results = confusion_matrix_metric.compute(
    predictions=predictions,
    references=true_labels
)

# The confusion matrix is a 2x2 array for binary classification:
# [[TN, FP],
#  [FN, TP]]
# Where:
# - TN: True Negatives
# - FP: False Positives
# - FN: False Negatives
# - TP: True Positives

print("Confusion Matrix (2x2):")
print(np.array(results["confusion_matrix"]))

# --- Visualization: Plot the confusion matrix as a heatmap ---
cm = np.array(results["confusion_matrix"])
fig, ax = plt.subplots(figsize=(4, 4))
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)

# Set axis labels and ticks
ax.set(
    xticks=[0, 1], yticks=[0, 1],
    xticklabels=["Predicted 0", "Predicted 1"],
    yticklabels=["Actual 0", "Actual 1"],
    ylabel="True label",
    xlabel="Predicted label",
    title="Confusion Matrix (2x2)"
)

# Annotate each cell with the count
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.show()

# --- Calculate and print metrics ---
TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

# Accuracy
accuracy = (TP + TN) / (TP + TN + FP + FN)
# Precision
precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
# Recall
recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
# F1 Score
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

print("\nMetrics:")
print(f"Accuracy:  {accuracy:.2f}   (Formula: (TP + TN) / (TP + TN + FP + FN))")
print(f"Precision: {precision:.2f}   (Formula: TP / (TP + FP))")
print(f"Recall:    {recall:.2f}   (Formula: TP / (TP + FN))")
print(f"F1 Score:  {f1:.2f}   (Formula: 2 * (Precision * Recall) / (Precision + Recall))")

print("\nExplanation:")
print("- True Negatives (TN): Correctly predicted 0s (top-left)")
print("- False Positives (FP): Incorrectly predicted 1s (top-right)")
print("- False Negatives (FN): Incorrectly predicted 0s (bottom-left)")
print("- True Positives (TP): Correctly predicted 1s (bottom-right)")
print("\nIn this example, the confusion matrix shows how many predictions fall into each category.")
