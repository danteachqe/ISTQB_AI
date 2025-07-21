import itertools
import pandas as pd
from allpairspy import AllPairs
import time

# --- 1. Define AI System Parameters ---
# Ensure at least five parameters
# Ensure enough options per parameter to achieve > 500 total combinations
parameters = {
    "weather": ["sunny", "rainy", "snowy", "foggy"],
    "road_type": ["highway", "urban", "rural", "off-road"],
    "traffic_density": ["low", "medium", "high", "jammed"],
    "sensor_quality": ["perfect", "good", "average", "poor", "faulty"],
    "vehicle_function": ["lane_keeping", "lane_change", "cruise_control", "emergency_braking", "parking_assist"],
    "lighting_conditions": ["daylight", "dusk", "night_lit", "night_unlit"] # Added a 6th parameter
}

# Calculate total theoretical combinations
# This ensures we meet the "at least five hundred possible combinations" requirement
total_theoretical_combinations = 1
for param_values in parameters.values():
    total_theoretical_combinations *= len(param_values)

print(f"--- AI System Configuration ---")
print(f"Parameters defined: {len(parameters)}")
print(f"🔢 Total Theoretical Combinations: {total_theoretical_combinations}")

# Check if the total theoretical combinations meet the minimum requirement
if total_theoretical_combinations < 500:
    print("\n⚠️ Warning: Total theoretical combinations are less than 500. Consider adding more options or parameters.")

# --- 2. Generate Pairwise Combinations ---
pairwise_combinations = list(AllPairs(parameters.values()))
pairwise_df = pd.DataFrame(pairwise_combinations, columns=parameters.keys())

print(f"\n--- Pairwise Testing Results ---")
print(f"✅ Pairwise Combinations Generated: {len(pairwise_combinations)}")

# --- 3. Compare Number of Combinations ---
reduction_percentage = 100 - (len(pairwise_combinations) / total_theoretical_combinations) * 100
print(f"📉 Reduction from exhaustive testing: {reduction_percentage:.2f}%")
print(f"Number of pairwise combinations tested: {len(pairwise_combinations)}")
print(f"Number required if all theoretically possible combinations were to be tested: {total_theoretical_combinations}")

# --- 4. Execute Tests for Pairwise Combinations (Simulated) ---
print("\n--- Executing Pairwise Tests (Simulated) ---")

def execute_ai_test(combination):
    """
    Simulates testing an AI-based system with a given set of parameters.
    In a real-world scenario, this would involve calling the AI system
    with these parameters and checking its behavior/output.
    """
    time.sleep(0.01) # Simulate some processing time

    # Convert dict_keys to a list to use .index()
    param_keys_list = list(parameters.keys())

    # Simulate a test outcome (e.g., success or failure based on some conditions)
    if combination[param_keys_list.index("sensor_quality")] == "faulty" and \
       combination[param_keys_list.index("traffic_density")] == "jammed":
        return "FAIL (Critical Scenario)"
    elif combination[param_keys_list.index("weather")] == "snowy" and \
         combination[param_keys_list.index("road_type")] == "off-road":
        return "WARNING (Challenging Conditions)"
    else:
        return "PASS"

test_results = []
for i, combination in enumerate(pairwise_combinations):
    result = execute_ai_test(combination)
    test_results.append({
        **{key: value for key, value in zip(parameters.keys(), combination)},
        "Test_Result": result
    })
    # print(f"Test {i+1}/{len(pairwise_combinations)}: {result}")

results_df = pd.DataFrame(test_results)

print(f"\n--- Pairwise Test Cases and Results ---")
print(results_df.to_string(index=False))

# --- Additional Analysis (Optional) ---
pass_count = results_df[results_df["Test_Result"] == "PASS"].shape[0]
fail_count = results_df[results_df["Test_Result"].str.startswith("FAIL")].shape[0]
warning_count = results_df[results_df["Test_Result"].str.startswith("WARNING")].shape[0]

print(f"\n--- Test Summary ---")
print(f"Total Pairwise Tests Executed: {len(pairwise_combinations)}")
print(f"Successful Tests (PASS): {pass_count}")
print(f"Failed Tests (FAIL): {fail_count}")
print(f"Tests with Warnings (WARNING): {warning_count}")