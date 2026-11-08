import pandas as pd

# Step 1: Load the dataset
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

# Step 5: Separate input features (X)
X = dataset.drop(columns=["Churn"])

# Step 6: Separate target variable (y)
y = dataset["Churn"]

# Step 7: Display the results
print("\nINPUT FEATURES (X)")
print(X.head())

print("\nTARGET VARIABLE (y)")
print(y.head())

print("\nFEATURE SHAPE:", X.shape)
print("TARGET SHAPE:", y.shape)

print("\nFEATURE COLUMNS:")
print(X.columns.tolist())

print("\nTARGET VALUES:")
print(y.value_counts())

print("\nFEATURE AND TARGET SEPARATION COMPLETED!")
