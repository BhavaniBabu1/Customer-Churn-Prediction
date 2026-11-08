# ==========================================
# CUSTOMER CHURN PREDICTION PROJECT
# STEP 17 - SECURITY AND SUPPORT ANALYSIS
# ==========================================

# 1. Import libraries
import pandas as pd
import matplotlib.pyplot as plt

# 2. Load dataset
dataset = pd.read_csv("dataset/telco_customer_churn.csv")

# 3. Analyze churn by Online Security
security_churn = pd.crosstab(
    dataset["OnlineSecurity"],
    dataset["Churn"],
    normalize="index"
) * 100

# 4. Analyze churn by Tech Support
support_churn = pd.crosstab(
    dataset["TechSupport"],
    dataset["Churn"],
    normalize="index"
) * 100

# 5. Display results
print("\n========== CHURN BY ONLINE SECURITY ==========")
print(security_churn.round(2))

print("\n========== CHURN BY TECH SUPPORT ==========")
print(support_churn.round(2))

# 6. Create two charts
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Online Security chart
security_churn.plot(
    kind="bar",
    ax=axes[0],
    color=["seagreen", "tomato"]
)

axes[0].set_title("Churn by Online Security")
axes[0].set_xlabel("Online Security")
axes[0].set_ylabel("Percentage of Customers")
axes[0].tick_params(axis="x", rotation=20)
axes[0].legend(title="Churn")

# Tech Support chart
support_churn.plot(
    kind="bar",
    ax=axes[1],
    color=["seagreen", "tomato"]
)

axes[1].set_title("Churn by Tech Support")
axes[1].set_xlabel("Tech Support")
axes[1].set_ylabel("Percentage of Customers")
axes[1].tick_params(axis="x", rotation=20)
axes[1].legend(title="Churn")

# 7. Adjust layout
plt.tight_layout()

# 8. Display charts
plt.show()

print("\nSECURITY AND SUPPORT ANALYSIS COMPLETED!")
