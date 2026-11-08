import pandas as pd

from sklearn.model_selection import train_test_split

# Step 1: Load dataset
dataset = pd.read_csv("dataset/telco_customer_churn.csv")

# Step 2: Convert TotalCharges into numeric format
dataset["TotalCharges"] = pd.to_numeric(
    dataset["TotalCharges"],
    errors="coerce"
)

# Step 3: Remove customerID
dataset = dataset.drop(columns=["customerID"])

# Step 4: Separate features and target
X = dataset.drop(columns=["Churn"])
y = dataset["Churn"].map({"No": 0, "Yes": 1})

# Step 5: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Step 6: Display dataset sizes
print("\nORIGINAL DATASET:", X.shape)

print("\nTRAINING FEATURES:", X_train.shape)
print("TESTING FEATURES:", X_test.shape)

print("\nTRAINING TARGET:", y_train.shape)
print("TESTING TARGET:", y_test.shape)

# Step 7: Check churn distribution
print("\nTRAINING CHURN DISTRIBUTION:")
print(y_train.value_counts(normalize=True).round(3) * 100)

print("\nTESTING CHURN DISTRIBUTION:")
print(y_test.value_counts(normalize=True).round(3) * 100)

print("\nTRAIN-TEST SPLIT COMPLETED!")
