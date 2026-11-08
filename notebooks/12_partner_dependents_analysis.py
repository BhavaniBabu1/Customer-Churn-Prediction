# ==========================================
# CUSTOMER CHURN PREDICTION PROJECT
# STEP 16 - PARTNER AND DEPENDENTS ANALYSIS
# ==========================================

# 1. Import libraries
import pandas as pd
import matplotlib.pyplot as plt

# 2. Load dataset
dataset = pd.read_csv("dataset/telco_customer_churn.csv")

# 3. Analyze churn by Partner
partner_churn = pd.crosstab(
    dataset["Partner"],
    dataset["Churn"],
    normalize="index"
) * 100

# 4. Analyze churn by Dependents
dependents_churn = pd.crosstab(
    dataset["Dependents"],
    dataset["Churn"],
    normalize="index"
) * 100

# 5. Display results
print("\n========== CHURN BY PARTNER ==========")
print(partner_churn)

print("\n========== CHURN BY DEPENDENTS ==========")
print(dependents_churn)

# 6. Create two charts
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Partner chart
partner_churn.plot(
    kind="bar",
    ax=axes[0],
    color=["seagreen", "tomato"]
)

axes[0].set_title("Customer Churn by Partner")
axes[0].set_xlabel("Has Partner?")
axes[0].set_ylabel("Percentage of Customers")
axes[0].tick_params(axis="x", rotation=0)
axes[0].legend(title="Churn")

# Dependents chart
dependents_churn.plot(
    kind="bar",
    ax=axes[1],
    color=["seagreen", "tomato"]
)

axes[1].set_title("Customer Churn by Dependents")
axes[1].set_xlabel("Has Dependents?")
axes[1].set_ylabel("Percentage of Customers")
axes[1].tick_params(axis="x", rotation=0)
axes[1].legend(title="Churn")

# 7. Adjust layout
plt.tight_layout()

# 8. Display charts
plt.show()

print("\nPARTNER AND DEPENDENTS ANALYSIS COMPLETED!")
