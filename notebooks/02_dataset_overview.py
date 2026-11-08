# ==========================================
# CUSTOMER CHURN PREDICTION PROJECT
# STEP 6 - DATASET OVERVIEW
# ==========================================

# 1. Import Pandas
import pandas as pd

# 2. Load the dataset
dataset = pd.read_csv("dataset/telco_customer_churn.csv")

# 3. Display basic dataset information
print("\n========== DATASET INFORMATION ==========")
dataset.info()

# 4. Display data types of all columns
print("\n========== COLUMN DATA TYPES ==========")
print(dataset.dtypes)

# 5. Display statistical summary
print("\n========== STATISTICAL SUMMARY ==========")
print(dataset.describe())

# 6. Display missing values in each column
print("\n========== MISSING VALUES ==========")
print(dataset.isnull().sum())
blank_total_charges = (
    dataset["TotalCharges"].astype(str).str.strip().eq("").sum()
)

print("Blank TotalCharges entries:", blank_total_charges)

# 7. Display number of unique values in each column
print("\n========== UNIQUE VALUES ==========")
print(dataset.nunique())

print("\nDATASET OVERVIEW COMPLETED SUCCESSFULLY!")