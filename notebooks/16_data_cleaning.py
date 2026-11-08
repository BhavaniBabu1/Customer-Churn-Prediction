import pandas as pd

# Load the original dataset
dataset = pd.read_csv("dataset/telco_customer_churn.csv")

print("\n========== DATA CLEANING ==========")

# Step 1: Convert TotalCharges from text to numeric
dataset["TotalCharges"] = pd.to_numeric(
    dataset["TotalCharges"],
    errors="coerce"
)

# Step 2: Check missing values before cleaning
print("\nMissing values before cleaning:")
print(dataset.isnull().sum())

# Step 3: Display customers with missing TotalCharges
print("\nCustomers with missing TotalCharges:")
print(
    dataset[dataset["TotalCharges"].isnull()][
        ["tenure", "MonthlyCharges", "TotalCharges"]
    ]
)

# Step 4: Check whether missing TotalCharges belongs to new customers
print("\nTenure values of customers with missing TotalCharges:")
print(
    dataset.loc[
        dataset["TotalCharges"].isnull(), "tenure"
    ].value_counts()
)

# Step 5: Replace missing TotalCharges with 0
# These records represent customers with zero months of tenure.
dataset["TotalCharges"] = dataset["TotalCharges"].fillna(0)

# Step 6: Check missing values after cleaning
print("\nMissing values after cleaning:")
print(dataset.isnull().sum())

print("\nDataset shape:", dataset.shape)

print("\n========== DATA CLEANING COMPLETED ==========")
