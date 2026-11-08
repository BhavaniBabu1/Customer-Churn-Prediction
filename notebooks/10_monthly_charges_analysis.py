# ==========================================
# CUSTOMER CHURN PREDICTION PROJECT
# STEP 14 - MONTHLY CHARGES ANALYSIS
# ==========================================

# 1. Import libraries
import pandas as pd
import matplotlib.pyplot as plt

# 2. Load dataset
dataset = pd.read_csv("dataset/telco_customer_churn.csv")

# 3. Create monthly charge groups
dataset["ChargesGroup"] = pd.cut(
    dataset["MonthlyCharges"],
    bins=[0, 35, 70, float("inf")],
    labels=["Low Charges", "Medium Charges", "High Charges"],
    include_lowest=True
)

# 4. Calculate churn percentage
charges_churn = pd.crosstab(
    dataset["ChargesGroup"],
    dataset["Churn"],
    normalize="index"
) * 100

# 5. Display results
print("\n========== CHURN BY MONTHLY CHARGES ==========")
print(charges_churn)

# 6. Create bar chart
charges_churn.plot(
    kind="bar",
    figsize=(10, 6),
    color=["seagreen", "tomato"]
)

# 7. Add chart labels
plt.title("Customer Churn by Monthly Charges")
plt.xlabel("Monthly Charges Category")
plt.ylabel("Percentage of Customers")
plt.xticks(rotation=0)
plt.legend(title="Churn")

# 8. Display chart
plt.tight_layout()
plt.show()

print("\nMONTHLY CHARGES ANALYSIS COMPLETED!")
