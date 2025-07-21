import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import random

# Define hardcoded housing dataset
data = [
    {"size": 1400, "bedrooms": 3, "age": 10, "garage": 1, "price": 245},
    {"size": 1600, "bedrooms": 3, "age": 5, "garage": 1, "price": 312},
    {"size": 1700, "bedrooms": 4, "age": 20, "garage": 0, "price": 279},
    {"size": 1875, "bedrooms": 3, "age": 15, "garage": 1, "price": 308},
    {"size": 1100, "bedrooms": 2, "age": 30, "garage": 0, "price": 199},
    {"size": 1550, "bedrooms": 3, "age": 7, "garage": 1, "price": 305},
    {"size": 2350, "bedrooms": 4, "age": 12, "garage": 1, "price": 420},
    {"size": 2450, "bedrooms": 4, "age": 8, "garage": 1, "price": 445},
    {"size": 1425, "bedrooms": 3, "age": 20, "garage": 0, "price": 229},
    {"size": 1700, "bedrooms": 4, "age": 10, "garage": 0, "price": 290},
    {"size": 1600, "bedrooms": 3, "age": 25, "garage": 1, "price": 280},
    {"size": 2000, "bedrooms": 4, "age": 10, "garage": 1, "price": 370},
    {"size": 2200, "bedrooms": 4, "age": 8, "garage": 1, "price": 390},
    {"size": 1250, "bedrooms": 2, "age": 40, "garage": 0, "price": 190},
    {"size": 1800, "bedrooms": 3, "age": 6, "garage": 1, "price": 325},
    {"size": 1900, "bedrooms": 3, "age": 4, "garage": 1, "price": 340},
    {"size": 2000, "bedrooms": 3, "age": 3, "garage": 1, "price": 355},
    {"size": 1450, "bedrooms": 2, "age": 15, "garage": 0, "price": 220},
    {"size": 1350, "bedrooms": 3, "age": 18, "garage": 1, "price": 240},
    {"size": 1600, "bedrooms": 3, "age": 9, "garage": 1, "price": 310},
    {"size": 2100, "bedrooms": 4, "age": 10, "garage": 1, "price": 380},
    {"size": 1150, "bedrooms": 2, "age": 30, "garage": 0, "price": 185},
    {"size": 1550, "bedrooms": 3, "age": 14, "garage": 1, "price": 295},
    {"size": 1950, "bedrooms": 4, "age": 7, "garage": 1, "price": 345},
    {"size": 1750, "bedrooms": 3, "age": 12, "garage": 1, "price": 300},
    {"size": 1300, "bedrooms": 2, "age": 20, "garage": 0, "price": 210}
]

df = pd.DataFrame(data)

# Split data
train_df = df.iloc[:21]
test_df = df.iloc[21:]

X_train = train_df.drop("price", axis=1)
y_train = train_df["price"]

X_test = test_df.drop("price", axis=1)
y_test = test_df["price"]

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
predictions = model.predict(X_test)

# Display test set predictions
results = pd.DataFrame({
    "Size": X_test["size"],
    "Bedrooms": X_test["bedrooms"],
    "Age": X_test["age"],
    "Garage": X_test["garage"],
    "Actual Price": y_test.values,
    "Predicted Price": predictions.round(2)
})

print("\n🏠 Test Set Predictions:")
print(results.to_string(index=False))

mse = mean_squared_error(y_test, predictions)
print(f"\n📉 Mean Squared Error on Test Set: {mse:.2f}")


# ───────────────────────────────
# Metamorphic Testing Section
# ───────────────────────────────
print("\n\n--- 🚀 Starting Metamorphic Testing ---")

def apply_transformation(data_row_series, feature_name, value):
    """Applies a simple additive transformation to a specific feature."""
    transformed_row = data_row_series.astype(float) # Ensure float dtype for consistency
    transformed_row[feature_name] += value
    # Ensure no negative values for physical attributes
    if feature_name in ['size', 'bedrooms', 'age']:
        transformed_row[feature_name] = max(0, transformed_row[feature_name])
    return transformed_row

def run_metamorphic_test(stc_data_row, mr_name, transform_func, expected_relation_desc,
                         expected_result_check_func, model_predictor, original_prediction):
    """
    Runs a single metamorphic test for a regression model.
    Returns a dictionary of results.
    """
    ftc_data_row_series = transform_func(stc_data_row)
    ftc_data_row_df = pd.DataFrame([ftc_data_row_series])

    ftc_predicted_value = model_predictor(ftc_data_row_df)[0]

    test_pass = expected_result_check_func(original_prediction, ftc_predicted_value)

    return {
        "MR": mr_name,
        "STC_Features": stc_data_row.to_dict(),
        "STC_Pred_Price": original_prediction,
        "FTC_Features": ftc_data_row_series.to_dict(),
        "FTC_Pred_Price": ftc_predicted_value,
        "Expected_Relation": expected_relation_desc,
        "Test_Pass": test_pass
    }

# --- Define Metamorphic Relations (MRs) and their checks ---

# MR1: Increase 'size' -> Price should increase
def mr1_transform(data_row):
    return apply_transformation(data_row, "size", 100) # Add 100 sqft
def mr1_check(original_pred, ftc_pred):
    return ftc_pred > original_pred

# MR2: Increase 'age' -> Price should decrease
def mr2_transform(data_row):
    return apply_transformation(data_row, "age", 5) # Add 5 years
def mr2_check(original_pred, ftc_pred):
    return ftc_pred < original_pred

# MR3: Duplicate input -> Price should be almost exactly the same (deterministic)
def mr3_transform(data_row):
    return data_row.astype(float) # Ensure float dtype for consistency, though no change in value
def mr3_check(original_pred, ftc_pred):
    return np.isclose(original_pred, ftc_pred, atol=0.01) # Small tolerance for floating point


# --- Execute Metamorphic Tests for each STC in X_test ---
all_mr_results = []

for i in range(len(X_test)):
    stc_data_row_series = X_test.iloc[i]
    original_prediction_stc = model.predict(pd.DataFrame([stc_data_row_series]))[0]

    print(f"\nEvaluating MRs for Source Test Case {i+1} (Original: {stc_data_row_series.to_dict()} -> ${original_prediction_stc:.2f})")

    all_mr_results.append(run_metamorphic_test(
        stc_data_row_series,
        "MR1: Increase 'size' (+100 sqft)",
        mr1_transform,
        "FTC Price > STC Price",
        mr1_check,
        model.predict,
        original_prediction_stc
    ))

    all_mr_results.append(run_metamorphic_test(
        stc_data_row_series,
        "MR2: Increase 'age' (+5 years)",
        mr2_transform,
        "FTC Price < STC Price",
        mr2_check,
        model.predict,
        original_prediction_stc
    ))

    all_mr_results.append(run_metamorphic_test(
        stc_data_row_series,
        "MR3: Duplicate Input (Deterministic Check)",
        mr3_transform,
        "FTC Price ≈ STC Price (exact match expected)",
        mr3_check,
        model.predict,
        original_prediction_stc
    ))

# --- Final Metamorphic Test Summary Report ---
print("\n\n--- ✨ Final Metamorphic Test Summary Report ---")
summary_df = pd.DataFrame(all_mr_results)

# Create compact feature string representations
summary_df['STC Features'] = summary_df['STC_Features'].apply(
    lambda x: f"S:{int(x['size'])}, B:{int(x['bedrooms'])}, A:{int(x['age'])}, G:{int(x['garage'])}"
)
summary_df['FTC Features'] = summary_df['FTC_Features'].apply(
    lambda x: f"S:{int(x['size'])}, B:{int(x['bedrooms'])}, A:{int(x['age'])}, G:{int(x['garage'])}"
)

# Format numerical columns
summary_df['STC Pred Price'] = summary_df['STC_Pred_Price'].map('${:,.2f}'.format)
summary_df['FTC Pred Price'] = summary_df['FTC_Pred_Price'].map('${:,.2f}'.format)
summary_df['Pass'] = summary_df['Test_Pass'].apply(lambda x: '✅ PASS' if x else '❌ FAIL')


# Select and reorder columns for final display
summary_df_display = summary_df[[
    "MR", "STC Features", "STC Pred Price",
    "FTC Features", "FTC Pred Price",
    "Expected_Relation", "Pass"
]].copy()

# Rename "Expected_Relation" for display
summary_df_display.rename(columns={"Expected_Relation": "Expected Behavior"}, inplace=True)

# Set display options for full table visibility
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 150) # Adjust for wider output

print(summary_df_display.to_string(index=False))

print("\n--- Metamorphic Testing Complete ---")