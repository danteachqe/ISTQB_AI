import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression # Back to Logistic Regression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import sys # For sys.exit()

# Import LIME components
from lime.lime_tabular import LimeTabularExplainer

# --- 1. Generate Synthetic Data Pool (Module-Level) ---
print("--- Generating Synthetic Module-Level Defect Data ---")

num_developers = 7
num_modules_per_dev = 30
total_modules = num_developers * num_modules_per_dev

developers = [f'Dev_{i+1}' for i in range(num_developers)]

# Simulate varying skill levels (lower skill -> higher defect proneness)
dev_skill_levels = {dev: np.random.uniform(0.3, 0.9) for dev in developers}
dev_skill_levels['Dev_1'] = 0.4 # Slightly lower skill
dev_skill_levels['Dev_2'] = 0.8 # Higher skill
dev_skill_levels['Dev_3'] = 0.3 # Even lower skill

data = []
for dev_id in developers:
    skill = dev_skill_levels[dev_id]
    for i in range(num_modules_per_dev):
        loc = int(np.random.normal(200, 80)) # Avg 200 LOC
        if loc < 50: loc = 50
        if loc > 600: loc = 600

        complexity = int(np.random.normal(15, 8)) # Avg 15 complexity
        if complexity < 5: complexity = 5
        if complexity > 50: complexity = 50

        num_comments = int(np.random.normal(loc * 0.1, loc * 0.05))
        if num_comments < 0: num_comments = 0

        num_revisions = int(np.random.normal(5, 3))
        if num_revisions < 1: num_revisions = 1

        unit_test_coverage = np.random.normal(0.85, 0.15)
        if unit_test_coverage < 0.05: unit_test_coverage = 0.05
        if unit_test_coverage > 1.0: unit_test_coverage = 1.0

        time_spent = int(np.random.normal(loc * 0.15, loc * 0.05))
        if time_spent < 1: time_spent = 1

        defect_prob = (
            (loc / 600.0 * 0.3) +
            (complexity / 50.0 * 0.3) +
            (1.0 - unit_test_coverage) * 0.2 +
            (num_revisions / 20.0 * 0.1) +
            (1.0 - skill) * 0.5
        )
        defect_prob = min(max(defect_prob, 0.01), 0.99)

        defects_introduced = 1 if np.random.rand() < defect_prob else 0

        data.append({
            'DeveloperID': dev_id,
            'CodeModule': f'Module_{dev_id}_{i+1}',
            'LinesOfCode': loc,
            'CyclomaticComplexity': complexity,
            'NumComments': num_comments,
            'NumRevisions': num_revisions,
            'UnitTestCoverage': unit_test_coverage,
            'TimeSpentOnModuleHours': time_spent,
            'DefectsIntroduced': defects_introduced,
            'DeveloperSkillLevel': skill # For internal generation, not a feature
        })

df = pd.DataFrame(data) # This 'df' is our module-level data for prediction
print(f"Generated dataset with {len(df)} code modules.")
print(df.head())
print("\nDefect Distribution (Module-level):\n", df['DefectsIntroduced'].value_counts())


# --- 2. Initial Performance Visuals Per Person ---
# (This section remains for the initial developer performance overview)
print("\n--- Initial Performance Visuals (Per Developer) ---")

dev_summary = df.groupby('DeveloperID').agg(
    TotalModules=('CodeModule', 'count'),
    TotalLOC=('LinesOfCode', 'sum'),
    TotalDefects=('DefectsIntroduced', 'sum'),
    AvgComplexity=('CyclomaticComplexity', 'mean'),
    AvgCoverage=('UnitTestCoverage', 'mean')
).reset_index()

dev_summary['DefectRatePerModule'] = dev_summary['TotalDefects'] / dev_summary['TotalModules']
dev_summary['DefectRatePer1kLOC'] = (dev_summary['TotalDefects'] / dev_summary['TotalLOC']) * 1000

print("\nDeveloper Summary Statistics:")
print(dev_summary.round(2))

# Visuals - DISPLAYED FIRST
plt.style.use('seaborn-v0_8-darkgrid')

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Plot 1: Total Defects per Developer
sns.barplot(x='DeveloperID', y='TotalDefects', data=dev_summary.sort_values('TotalDefects', ascending=False), ax=axes[0], palette='viridis')
axes[0].set_title('Total Defects Introduced by Developer')
axes[0].set_ylabel('Number of Defects')
axes[0].set_xlabel('Developer')
axes[0].tick_params(axis='x', rotation=45)

# Plot 2: Defect Rate per Module per Developer
sns.barplot(x='DeveloperID', y='DefectRatePerModule', data=dev_summary.sort_values('DefectRatePerModule', ascending=False), ax=axes[1], palette='plasma')
axes[1].set_title('Defect Rate per Module by Developer')
axes[1].set_ylabel('Defects per Module')
axes[1].set_xlabel('Developer')
axes[1].tick_params(axis='x', rotation=45)

# Plot 3: Defect Rate per 1000 LOC per Developer
sns.barplot(x='DeveloperID', y='DefectRatePer1kLOC', data=dev_summary.sort_values('DefectRatePer1kLOC', ascending=False), ax=axes[2], palette='cividis')
axes[2].set_title('Defect Rate per 1000 LOC by Developer')
axes[2].set_ylabel('Defects per 1000 LOC')
axes[2].set_xlabel('Developer')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.suptitle('Developer Performance Overview (Pre-Modeling)', y=1.02, fontsize=16)
plt.show() # Display the graphs NOW


# --- 3. Build Defect Probability Prediction Models (Logistic Regression) ---
print("\n--- Building Module Defect Probability Models ---")

# Define features and target (module-level again)
target = 'DefectsIntroduced'

# Model 1: Core Features
features_model1 = ['LinesOfCode', 'CyclomaticComplexity', 'UnitTestCoverage', 'TimeSpentOnModuleHours']

# Model 2: Expanded Features (including all generated numerical features)
features_model2 = ['LinesOfCode', 'CyclomaticComplexity', 'NumComments', 'NumRevisions', 'UnitTestCoverage', 'TimeSpentOnModuleHours']

models = {
    "Model 1 (Core Features)": features_model1,
    "Model 2 (Expanded Features)": features_model2
}

# Prepare data for modeling (scaling is important for Logistic Regression)
X = df[features_model2] # Use all features for X, then select subset later for models
y = df[target]

# Scale numerical features - scaler must be fitted on the FULL X data (before train/test split)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
print(f"\nTraining data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Train and evaluate each model
final_model = None # To store the last trained model (Model 2) for interactive prediction
final_features = [] # To store features of the last trained model

for model_name, features in models.items():
    print(f"\n----- {model_name} -----")
    
    # Select features for the current model
    X_train_model = X_train[features]
    X_test_model = X_test[features]

    # Initialize and train Logistic Regression model
    log_reg_model = LogisticRegression(random_state=42, solver='liblinear')
    log_reg_model.fit(X_train_model, y_train)

    # Make predictions
    y_pred = log_reg_model.predict(X_test_model)
    
    # Evaluate model performance
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # Feature Coefficients
    print("\nFeature Coefficients (Scaled Data):")
    coef_df = pd.DataFrame({
        'Feature': features,
        'Coefficient': log_reg_model.coef_[0]
    }).sort_values(by='Coefficient', ascending=False)
    print(coef_df)
    print(f"\nInterpretation: Positive coefficient increases log-odds of defect probability. Negative coefficient decreases log-odds.")

    # Store the last trained model for interactive prediction
    final_model = log_reg_model
    final_features = features


# --- 4. Dynamic Prediction for New Code Module + LIME Explanation ---
print("\n--- Interactive Defect Likelihood Prediction + LIME Explanation ---")
print("Enter the characteristics for a new code module to predict its defect likelihood.")
print("Type 'exit' or 'quit' at any prompt to stop.")

def get_numeric_input(prompt, type_func=int, min_val=None, max_val=None):
    while True:
        user_input = input(prompt).strip().lower()
        if user_input in ['exit', 'quit']:
            print("Exiting prediction system.")
            sys.exit()
        try:
            value = type_func(user_input)
            if min_val is not None and value < min_val:
                print(f"Input must be at least {min_val}. Please try again.")
            elif max_val is not None and value > max_val:
                print(f"Input must be at most {max_val}. Please try again.")
            else:
                return value
        except ValueError:
            print("Invalid input. Please enter a valid number.")

# Features used by the final model (Model 2 in this case)
required_features_for_prediction = [
    ('LinesOfCode', 'Enter Lines of Code (e.g., 250): ', float, 50, 60000),
    ('CyclomaticComplexity', 'Enter Cyclomatic Complexity (e.g., 20): ', int, 5, 50),
    ('NumComments', 'Enter Number of Comments (e.g., 40): ', int, 0, 100),
    ('NumRevisions', 'Enter Number of Revisions (e.g., 7): ', int, 1, 20),
    ('UnitTestCoverage', 'Enter Unit Test Coverage (0.0 to 1.0, e.g., 0.75): ', float, 0.0, 1.0),
    ('TimeSpentOnModuleHours', 'Enter Time Spent on Module (hours, e.g., 35): ', int, 1, 80),
]

new_module_values = {}
for feat_name, prompt, type_func, min_val, max_val in required_features_for_prediction:
    new_module_values[feat_name] = get_numeric_input(prompt, type_func, min_val, max_val)

new_module_df = pd.DataFrame([new_module_values])

print("\nYour input for the new module:")
print(new_module_df)

# Scale the new data using the SAME SCALER fitted on training data
# Note: scaler was fitted on X (all features_model2), so we need to use all those for transform
# even if a model used a subset. LIME needs the full feature space used by the scaler's training data.
new_module_for_scaler = new_module_df[features_model2] # Ensure this aligns with what scaler was fitted on
new_module_scaled_for_pred = pd.DataFrame(
    scaler.transform(new_module_for_scaler),
    columns=features_model2 # Columns should match what scaler expects
)

# Predict probabilities using the features the final model was trained on
# This uses X_train_model, which was `features_model2` if Model 2 was the last trained
predicted_proba = final_model.predict_proba(new_module_scaled_for_pred[final_features])[:, 1][0]
predicted_class = final_model.predict(new_module_scaled_for_pred[final_features])[0]

likelihood_category = ""
if predicted_proba < 0.2:
    likelihood_category = "Very Low"
elif predicted_proba < 0.4:
    likelihood_category = "Low"
elif predicted_proba < 0.6:
    likelihood_category = "Medium"
elif predicted_proba < 0.8:
    likelihood_category = "High"
else:
    likelihood_category = "Very High"

print(f"\n--- Prediction Results for Your New Module ---")
print(f"Predicted Defect Probability: {predicted_proba:.4f}")
print(f"Predicted Defect Class (0=No Defect, 1=Defect): {predicted_class}")
print(f"Likelihood Category: {likelihood_category}")
print("---------------------------------------------")

# --- LIME Explainability for this SPECIFIC prediction ---
print("\n--- LIME Explanation: How these inputs influenced the prediction ---")

# Initialize LIME explainer
# Use X_train.values (unscaled training data) and feature_names from original data
explainer = LimeTabularExplainer(
    training_data=X_train.values, # LIME uses unscaled data for understanding feature ranges
    feature_names=X.columns.tolist(), # All original feature names
    class_names=['No Defect', 'Defect'], # Output classes
    mode='classification' # We are predicting a binary class
)

# Get the original (unscaled) input values for LIME's explanation text
unscaled_input_for_lime = new_module_df.iloc[0].values

# Explain the instance
# predict_proba needs to be a function that takes a 2D numpy array and returns probabilities
explanation = explainer.explain_instance(
    data_row=unscaled_input_for_lime, # LIME needs unscaled data for its internal perturbations
    predict_fn=final_model.predict_proba, # Function to predict probabilities
    num_features=len(final_features), # Show all features used by the model
    top_labels=1 # Only explain the predicted class
)

print("\nHow Each Feature Influences This Prediction (LIME):")

# Get explanation as a list of (feature_string, weight) tuples
explanation_list = explanation.as_list()

# Sort the features by the absolute value of their influence (descending) for readability
explanation_list.sort(key=lambda x: abs(x[1]), reverse=True)

for feature_string, weight in explanation_list:
    # Extract the original feature name from LIME's potentially binned string
    # e.g., 'LinesOfCode <= 200.00' -> 'LinesOfCode'
    original_feature_name = feature_string.split(' ')[0]

    # Get the actual value from the user's input for context
    feature_value = new_module_values.get(original_feature_name, 'N/A')

    influence_type = "positively increases defect likelihood" if weight > 0 else "negatively decreases defect likelihood"
    
    # Determine the strength of influence
    abs_weight = abs(weight)
    strength = ""
    if abs_weight > 0.1:
        strength = "strongly"
    elif abs_weight > 0.05:
        strength = "moderately"
    elif abs_weight > 0.01:
        strength = "slightly"
    else:
        strength = "minimally"

    print(f"  - The '{original_feature_name}' (value: {feature_value}) {strength} {influence_type} by approx. {abs_weight:.4f}")
print("------------------------------------------------------------------")

print("\n--- Defect Prediction System Complete ---")