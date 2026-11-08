# ==========================================
# CUSTOMER CHURN PREDICTION PROJECT
# STEP 13 - TENURE ANALYSIS
# ==========================================

# 1. Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Load dataset
dataset = pd.read_csv("dataset/telco_customer_churn.csv")

# 3. Create tenure groups
def tenure_group(months):
    if months <= 12:
        return "0-12 Months"
    elif months <= 24:
        return "13-24 Months"
    elif months <= 48:
        return "25-48 Months"
    else:
        return "49+ Months"

# 4. Apply tenure grouping
dataset["TenureGroup"] = dataset["tenure"].apply(tenure_group)

# 5. Calculate churn percentage
tenure_churn = pd.crosstab(
    dataset["TenureGroup"],
    dataset["Churn"],
    normalize="index"
) * 100

# 6. Display results
print("\n========== CHURN BY TENURE ==========")
print(tenure_churn)

# 7. Create visualization
tenure_churn.plot(
    kind="bar",
    figsize=(10, 6),
    color=["seagreen", "tomato"]
)

# 8. Add chart labels
plt.title("Customer Churn by Tenure")
plt.xlabel("Customer Tenure")
plt.ylabel("Percentage of Customers")
plt.xticks(rotation=0)
plt.legend(title="Churn")

# 9. Display chart
plt.tight_layout()
plt.show()

print("\nTENURE ANALYSIS COMPLETED!")
