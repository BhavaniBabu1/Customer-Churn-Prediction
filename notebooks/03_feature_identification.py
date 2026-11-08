# ==========================================
# CUSTOMER CHURN PREDICTION PROJECT
# STEP 7 - FEATURE IDENTIFICATION
# ==========================================

# 1. Import Pandas
import pandas as pd

# 2. Load the dataset
dataset = pd.read_csv("dataset/telco_customer_churn.csv")

# 3. Identify numerical columns
numerical_columns = dataset.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

# 4. Identify categorical columns
categorical_columns = dataset.select_dtypes(
    include=["object"]
).columns.tolist()

# 5. Display numerical columns
print("\n========== NUMERICAL COLUMNS ==========")
print(numerical_columns)

# 6. Display categorical columns
print("\n========== CATEGORICAL COLUMNS ==========")
print(categorical_columns)

# 7. Display the number of columns in each group
print("\n========== COLUMN COUNTS ==========")
print("Numerical columns:", len(numerical_columns))
print("Categorical columns:", len(categorical_columns))

print("\nFEATURE IDENTIFICATION COMPLETED!")
