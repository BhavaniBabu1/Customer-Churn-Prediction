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

# Step 5: Separate features and target
X = dataset.drop(columns=["Churn"])
y = dataset["Churn"]

# Step 6: Encode the target variable
y = y.map({"No": 0, "Yes": 1})

# Step 7: Display encoded values
print("\nENCODED TARGET VALUES")
print(y.head(10))

print("\nTARGET VALUE COUNTS")
print(y.value_counts())

print("\nMISSING VALUES IN TARGET:", y.isnull().sum())

print("\nTARGET ENCODING COMPLETED!")
