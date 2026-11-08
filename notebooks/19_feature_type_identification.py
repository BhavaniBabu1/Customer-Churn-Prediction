import pandas as pd

# Step 1: Load the dataset
dataset = pd.read_csv("dataset/telco_customer_churn.csv")

# Step 2: Convert TotalCharges into numeric format
dataset["TotalCharges"] = pd.to_numeric(
    dataset["TotalCharges"],
    errors="coerce"
)

# Step 3: Replace missing TotalCharges with 0
dataset["TotalCharges"] = dataset["TotalCharges"].fillna(0)

# Step 4: Remove customerID
dataset = dataset.drop(columns=["customerID"])

# Step 5: Separate features and target
X = dataset.drop(columns=["Churn"])
y = dataset["Churn"].map({"No": 0, "Yes": 1})

# Step 6: Identify numerical features
numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

# Step 7: Identify categorical features
categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

# Step 8: Display results
print("\nNUMERICAL FEATURES:")
print(numerical_features)

print("\nCATEGORICAL FEATURES:")
print(categorical_features)

print("\nNUMBER OF NUMERICAL FEATURES:", len(numerical_features))
print("NUMBER OF CATEGORICAL FEATURES:", len(categorical_features))

print("\nFEATURE TYPE IDENTIFICATION COMPLETED!")
