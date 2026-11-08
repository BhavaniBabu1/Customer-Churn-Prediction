import pandas as pd
from sklearn.preprocessing import StandardScaler

# Step 1: Load dataset
dataset = pd.read_csv("dataset/telco_customer_churn.csv")

# Step 2: Convert TotalCharges into numeric format
dataset["TotalCharges"] = pd.to_numeric(
    dataset["TotalCharges"],
    errors="coerce"
)

# Step 3: Handle missing TotalCharges
dataset["TotalCharges"] = dataset["TotalCharges"].fillna(0)

# Step 4: Remove customerID
dataset = dataset.drop(columns=["customerID"])

# Step 5: Separate numerical features
numerical_features = dataset.select_dtypes(
    include=["int64", "float64"]
).drop(columns=["SeniorCitizen"]).columns.tolist()

# Step 6: Display original numerical data
print("\nORIGINAL NUMERICAL DATA:")
print(dataset[numerical_features].head())

# Step 7: Create StandardScaler
scaler = StandardScaler()

# Step 8: Scale numerical features
scaled_values = scaler.fit_transform(
    dataset[numerical_features]
)

# Step 9: Convert scaled values into a DataFrame
scaled_data = pd.DataFrame(
    scaled_values,
    columns=numerical_features
)

# Step 10: Display scaled data
print("\nSCALED NUMERICAL DATA:")
print(scaled_data.head())

# Step 11: Check means after scaling
print("\nMEAN AFTER SCALING:")
print(scaled_data.mean().round(2))

# Step 12: Check standard deviations
print("\nSTANDARD DEVIATION AFTER SCALING:")
print(scaled_data.std().round(2))

print("\nFEATURE SCALING COMPLETED!")
