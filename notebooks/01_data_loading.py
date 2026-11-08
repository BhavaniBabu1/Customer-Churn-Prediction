
# ==========================================
# CUSTOMER CHURN PREDICTION PROJECT
# STEP 5 - LOADING THE DATASET
# ==========================================

# 1. Import Pandas library
import pandas as pd

# 2. Load the CSV dataset
dataset = pd.read_csv("dataset/telco_customer_churn.csv")

# 3. Display the first five records
print("\n FIRST FIVE CUSTOMER RECORDS")
print(dataset.head())

# 4. Display the shape of the dataset
print("\n DATASET SHAPE")
print(dataset.shape)

# 5. Display the column names
print("\n DATASET COLUMNS")
print(dataset.columns)

# 6. Display the last five records
print("\n LAST FIVE CUSTOMER RECORDS")
print(dataset.tail())

print("\n DATASET LOADED SUCCESSFULLY!")
