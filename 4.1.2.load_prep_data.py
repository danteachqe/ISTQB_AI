import pandas as pd
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

# ---------------

# 4.1.2 Hands-On Exercise: Data Preparation for ML
# Practical Example: Customer Segmentation for a Financial Institution
# ---------------------------------------------------------------
# This script demonstrates the essential data preparation steps for supervised learning.
# We simulate a real-world scenario where a financial institution wants to classify customers
# into three segments (e.g., low, medium, high value) based on demographic and financial features.
# The prepared dataset will be used for future ML model training and evaluation.
# ---------------------------------------------------------------

# Step 1: Generate a synthetic, complex, and unbalanced classification dataset
# - n_samples: Number of samples (rows)
# - n_features: Total number of features (columns)
# - n_informative: Number of informative features
# - n_redundant: Number of redundant features
# - n_classes: Number of target classes
# - weights: Class imbalance (e.g., 70% class 0, 20% class 1, 10% class 2)
# - flip_y: Randomly flip labels to introduce noise
# - random_state: Seed for reproducibility
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

# Step 2: Feature scaling (standardization)
# - StandardScaler transforms features to have zero mean and unit variance
# - This is important for many ML algorithms to perform optimally
print("Scaling features...")
scaler = StandardScaler()
X = scaler.fit_transform(X)
print(f"Features scaled. Shape: {X.shape}")


# Step 3: (Optional) Convert to pandas DataFrame for easier inspection and future use
# - Not strictly required, but helpful for EDA and saving to file
# Assign realistic feature names for demonstration
feature_names = [
    "age", "income", "education_years", "num_children", "credit_score",
    "account_balance", "num_transactions", "employment_years", "debt_ratio", "region_code"
]
df_X = pd.DataFrame(X, columns=feature_names)
df_y = pd.Series(y, name="target")

# Step 4: Output prepared data shapes
print(f"Prepared features shape: {df_X.shape}")
print(f"Prepared labels shape: {df_y.shape}")


# Step 5: (Optional) Save prepared data for future use
# df_X.to_csv("prepared_features.csv", index=False)
# df_y.to_csv("prepared_labels.csv", index=False)

# Step 6: Data exploration and summary statistics
print("\n=== Data Summary ===")
print("Feature summary statistics:")
print(df_X.describe())

print("\nClass distribution:")
print(df_y.value_counts(normalize=True).rename('proportion'))

# Step 7: Visualize feature distributions (histograms)
import matplotlib.pyplot as plt
df_X.hist(bins=20, figsize=(12, 8))
plt.suptitle("Feature Distributions")
plt.tight_layout()
plt.show()

# Step 8: Visualize class distribution (bar plot)
df_y.value_counts().sort_index().plot(kind='bar')
plt.title("Class Distribution")
plt.xlabel("Class label")
plt.ylabel("Count")
plt.show()

# The dataset is now ready for splitting and model training in future exercises.