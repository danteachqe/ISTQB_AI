import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# === Step 1: Generate synthetic dataset ===
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=6,
    n_redundant=2,
    n_classes=3,
    weights=[0.7, 0.2, 0.1],
    flip_y=0.01,
    random_state=42
)

feature_names = [
    "age", "income", "education_years", "num_children", "credit_score",
    "account_balance", "num_transactions", "employment_years", "debt_ratio", "region_code"
]
df_X = pd.DataFrame(X, columns=feature_names)
df_y = pd.Series(y, name="target")

# === Step 2: Simulate Data Issues ===

## 2.1 Introduce missing values
missing_fraction = 0.02
cols_with_missing = ['income', 'credit_score']
total_missing = 0
for col in cols_with_missing:
    n_missing = int(missing_fraction * len(df_X))
    missing_indices = df_X.sample(n=n_missing, random_state=42).index
    df_X.loc[missing_indices, col] = np.nan
    print(f"🔸 Introduced {n_missing} missing values in column: {col}")
    total_missing += n_missing

## 2.2 Add duplicates
df_X = pd.concat([df_X, df_X.iloc[:5]], ignore_index=True)
df_y = pd.concat([df_y, df_y.iloc[:5]], ignore_index=True)
print("🔸 Added 5 duplicate rows")

## 2.3 Add outliers
df_X.loc[0, 'account_balance'] = 1e6  # outlier
df_X.loc[1, 'debt_ratio'] = 999       # outlier
print("🔸 Added 2 outliers: account_balance (1e6), debt_ratio (999)")

# === Step 3: Data Cleaning ===

## 3.1 Drop duplicates
before_dedup = df_X.shape[0]
df_X = df_X.drop_duplicates()
df_y = df_y.loc[df_X.index]
dropped_dupes = before_dedup - df_X.shape[0]
print(f"✅ Dropped {dropped_dupes} duplicate rows")

## 3.2 Fill missing values with median
missing_before = df_X.isna().sum().sum()
df_X.fillna(df_X.median(), inplace=True)
missing_after = df_X.isna().sum().sum()
print(f"✅ Filled {missing_before - missing_after} missing values using median")

## 3.3 Remove outliers using IQR
def remove_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    mask = (df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)
    removed = (~mask).sum()
    return df[mask], removed

total_outliers = 0
for col in ['account_balance', 'debt_ratio']:
    before = df_X.shape[0]
    df_X, removed = remove_outliers(df_X, col)
    df_y = df_y.loc[df_X.index]  # keep labels aligned
    total_outliers += removed
    print(f"✅ Removed {removed} outliers from column: {col}")

# === Step 4: Feature Scaling ===
scaler = StandardScaler()
df_X_scaled = pd.DataFrame(scaler.fit_transform(df_X), columns=df_X.columns)

# === Step 5: Summary ===
print("\n📊 Final Dataset Summary:")
print(f"🔹 Rows: {df_X_scaled.shape[0]}")
print(f"🔹 Columns: {df_X_scaled.shape[1]}")
print(f"🔹 Total corrections made:")
print(f"   - Missing values filled: {total_missing}")
print(f"   - Duplicate rows dropped: {dropped_dupes}")
print(f"   - Outliers removed: {total_outliers}")

# === Step 6: Visuals ===
df_X_scaled.hist(bins=20, figsize=(12, 8))
plt.suptitle("Cleaned & Scaled Feature Distributions")
plt.tight_layout()
plt.show()

df_y.value_counts().sort_index().plot(kind='bar')
plt.title("Class Distribution")
plt.xlabel("Class Label")
plt.ylabel("Count")
plt.show()
