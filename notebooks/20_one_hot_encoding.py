import pandas as pd

# Step 1: Load dataset
dataset = pd.read_csv("dataset/telco_customer_churn.csv")

# Step 2: Convert TotalCharges into numeric format
dataset["TotalCharges"] = pd.to_numeric(
    dataset["TotalCharges"],
    errors="coerce"
)

# Step 3: Handle missing values
dataset["TotalCharges"] = dataset["TotalCharges"].fillna(0)

# Step 4: Remove customerID
dataset = dataset.drop(columns=["customerID"])

# Step 5: Separate features and target
X = dataset.drop(columns=["Churn"])
y = dataset["Churn"].map({"No": 0, "Yes": 1})

# Step 6: Identify categorical features
categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

# Step 7: Apply One-Hot Encoding
X_encoded = pd.get_dummies(
    X,
    columns=categorical_features,
    dtype=int
)

# Step 8: Display results
print("\nORIGINAL FEATURE DATA:")
print(X.head())

print("\nENCODED FEATURE DATA:")
print(X_encoded.head())

print("\nORIGINAL SHAPE:", X.shape)
print("ENCODED SHAPE:", X_encoded.shape)

print("\nENCODED COLUMN NAMES:")
print(X_encoded.columns.tolist())

print("\nONE-HOT ENCODING COMPLETED!")

