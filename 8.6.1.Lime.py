import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from lime.lime_tabular import LimeTabularExplainer

# Define hardcoded housing dataset (26 rows total, ensuring enough for 21 train, 5 test)
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
    {"size": 1150, "bedrooms": 2, "age": 30, "garage": 0, "price": 185}, # Test Case 1
    {"size": 1550, "bedrooms": 3, "age": 14, "garage": 1, "price": 295}, # Test Case 2
    {"size": 1950, "bedrooms": 4, "age": 7, "garage": 1, "price": 345},  # Test Case 3
    {"size": 1750, "bedrooms": 3, "age": 12, "garage": 1, "price": 300}, # Test Case 4
    {"size": 1300, "bedrooms": 2, "age": 20, "garage": 0, "price": 210}  # Test Case 5
]

# Convert to DataFrame
df = pd.DataFrame(data)

# Split data: first 21 rows = train, last 5 = test
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

# Display results
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

# Calculate error
mse = mean_squared_error(y_test, predictions)
print(f"\n📉 Mean Squared Error on Test Set: {mse:.2f}")

# -----------------------------------------------
# LIME Explainability Section (Enhanced Readability)
# -----------------------------------------------
print("\n\n--- 🔍 Generating Model Explanations using LIME ---")

feature_names = X_train.columns.tolist()

explainer = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=feature_names,
    class_names=['Price'],
    mode='regression'
)

# Choose a few instances from the test set to explain
# Explaining the first 3 test cases.
examples_to_explain_indices = [0, 1, 2]

print("\n")
for i in examples_to_explain_indices:
    instance = X_test.iloc[i].values
    predicted_price = model.predict(X_test.iloc[[i]])[0]

    print(f"--- Explaining Prediction for House {i+1} ---")
    print(f"Features: {X_test.iloc[i].to_dict()}")
    print(f"Predicted Price: ${predicted_price:.2f}")

    explanation = explainer.explain_instance(
        data_row=instance,
        predict_fn=model.predict,
        num_features=len(feature_names) # Show all features
    )

    print("\nHow Each Feature Influences This Prediction:")

    # Get explanation as a list of (feature, weight) tuples
    explanation_list = explanation.as_list()

    # Sort the features by the absolute value of their influence (descending)
    explanation_list.sort(key=lambda x: abs(x[1]), reverse=True)

    # Dictionary to map LIME's potentially binned feature names back to original names
    original_feature_values = X_test.iloc[i].to_dict()

    for feature_string, weight in explanation_list:
        # Extract the original feature name from LIME's feature_string
        # This handles cases like 'size <= 1450.00' to get 'size'
        original_feature_name = feature_string.split(' ')[0]

        # Safely get the value from the original DataFrame using the original feature name
        feature_value = original_feature_values.get(original_feature_name, 'N/A')

        influence_type = "positively influences" if weight > 0 else "negatively influences"
        # Determine the strength of influence for a more descriptive statement
        abs_weight = abs(weight)
        strength = ""
        if abs_weight > 0.1:
            strength = "strongly"
        elif abs_weight > 0.01:
            strength = "moderately"
        elif abs_weight > 0.001:
            strength = "slightly"
        else:
            strength = "minimally"


        print(f"  - The '{original_feature_name}' (value: {feature_value}) {strength} {influence_type} the price by approx. ${abs_weight:.2f}")
    print("-" * 60)
    print("\n")

print("--- LIME Explainability Complete ---")