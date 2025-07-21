import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from scipy.stats import f_oneway

# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = pd.Categorical.from_codes(iris.target, iris.target_names)


# Head
print("=== Head of Dataset ===")
print(df.head(), "\n")

# Missing data check
print("=== Missing Values Per Column ===")
print(df.isnull().sum(), "\n")

# Summary statistics
print("=== Summary Statistics ===")
print(df.describe(), "\n")

# Class distribution
print("=== Class Distribution ===")
print(df['target'].value_counts(), "\n")

# Pairplot
sns.pairplot(df, hue='target', diag_kind='hist')
plt.suptitle("Iris Pairplot by Class", y=1.02)
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.drop(columns='target').corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# Boxplots by class
for column in df.columns[:-1]:  # exclude target
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='target', y=column, data=df)
    plt.title(f"Boxplot of {column} by Class")
    plt.tight_layout()
    plt.show()


print("\n=== Bias & Imbalance Checks ===")

# 1. Check for class imbalance
print("\n[1] Class Distribution (%):")
class_dist = df['target'].value_counts(normalize=True) * 100
print(class_dist)

# 2. Check feature mean differences across classes
print("\n[2] Feature Means by Class:")
feature_means = df.groupby('target', observed=True).mean()
print(feature_means)

# 3. Check for statistical difference using ANOVA (per feature)
print("\n[3] ANOVA Test Results (Do classes differ significantly per feature?)")
for col in df.columns[:-1]:
    groups = [df[df['target'] == target][col] for target in df['target'].unique()]
    f_stat, p_val = f_oneway(*groups)
    print(f"{col}: F={f_stat:.2f}, p={p_val:.4f} {'(Significant)' if p_val < 0.05 else '(Not Significant)'}")
