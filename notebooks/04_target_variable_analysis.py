# ==========================================
# CUSTOMER CHURN PREDICTION PROJECT
# STEP 8 - TARGET VARIABLE ANALYSIS
# ==========================================

import pandas as pd

# Load dataset
dataset = pd.read_csv("dataset/telco_customer_churn.csv")

# Count customers in each churn category
print("\n========== CHURN COUNTS ==========")
print(dataset["Churn"].value_counts())

# Calculate churn percentages
print("\n========== CHURN PERCENTAGES ==========")
print(dataset["Churn"].value_counts(normalize=True) * 100)

# Store the counts
churn_counts = dataset["Churn"].value_counts()

# Display total customers
print("\n========== TOTAL CUSTOMERS ==========")
print("Total customers:", len(dataset))

# Display customers who left
print("\nCustomers who left:", churn_counts["Yes"])

# Display customers who stayed
print("Customers who stayed:", churn_counts["No"])

print("\nTARGET VARIABLE ANALYSIS COMPLETED!")
