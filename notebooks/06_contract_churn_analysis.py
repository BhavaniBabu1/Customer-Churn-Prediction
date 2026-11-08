# ==========================================
# CUSTOMER CHURN PREDICTION PROJECT
# STEP 10 - CONTRACT CHURN ANALYSIS
# ==========================================

# 1. Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Load dataset
dataset = pd.read_csv("dataset/telco_customer_churn.csv")

# 3. Calculate churn percentage for each contract
contract_churn = pd.crosstab(
    dataset["Contract"],
    dataset["Churn"],
    normalize="index"
) * 100

print("\n========== CHURN PERCENTAGE BY CONTRACT ==========")
print(contract_churn)

# 4. Create a bar chart
contract_churn.plot(
    kind="bar",
    figsize=(10, 6),
    color=["seagreen", "tomato"]
)

# 5. Add chart labels
plt.title("Customer Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Percentage of Customers")
plt.xticks(rotation=0)
plt.legend(title="Churn")

# 6. Display chart
plt.tight_layout()
plt.show()

print("\nCONTRACT ANALYSIS COMPLETED!")