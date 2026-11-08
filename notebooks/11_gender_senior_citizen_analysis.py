# ==========================================
# CUSTOMER CHURN PREDICTION PROJECT
# STEP 15 - GENDER AND SENIOR CITIZEN ANALYSIS
# ==========================================

# 1. Import libraries
import pandas as pd
import matplotlib.pyplot as plt

# 2. Load dataset
dataset = pd.read_csv("dataset/telco_customer_churn.csv")

# 3. Analyze churn by gender
gender_churn = pd.crosstab(
    dataset["gender"],
    dataset["Churn"],
    normalize="index"
) * 100

# 4. Analyze churn by senior citizen status
dataset["SeniorCitizenGroup"] = dataset["SeniorCitizen"].map({
    0: "Not Senior Citizen",
    1: "Senior Citizen"
})

senior_churn = pd.crosstab(
    dataset["SeniorCitizenGroup"],
    dataset["Churn"],
    normalize="index"
) * 100

# 5. Display results
print("\n========== CHURN BY GENDER ==========")
print(gender_churn)

print("\n========== CHURN BY SENIOR CITIZEN STATUS ==========")
print(senior_churn)

# 6. Create two charts
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Gender chart
gender_churn.plot(
    kind="bar",
    ax=axes[0],
    color=["seagreen", "tomato"]
)

axes[0].set_title("Customer Churn by Gender")
axes[0].set_xlabel("Gender")
axes[0].set_ylabel("Percentage of Customers")
axes[0].tick_params(axis="x", rotation=0)
axes[0].legend(title="Churn")

# Senior citizen chart
senior_churn.plot(
    kind="bar",
    ax=axes[1],
    color=["seagreen", "tomato"]
)

axes[1].set_title("Customer Churn by Senior Citizen Status")
axes[1].set_xlabel("Customer Category")
axes[1].set_ylabel("Percentage of Customers")
axes[1].tick_params(axis="x", rotation=10)
axes[1].legend(title="Churn")

# 7. Adjust layout and display charts
plt.tight_layout()
plt.show()

print("\nGENDER AND SENIOR CITIZEN ANALYSIS COMPLETED!")
