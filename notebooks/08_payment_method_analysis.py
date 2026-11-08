# ==========================================
# CUSTOMER CHURN PREDICTION PROJECT
# STEP 12 - PAYMENT METHOD ANALYSIS
# ==========================================

# 1. Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Load dataset
dataset = pd.read_csv("dataset/telco_customer_churn.csv")

# 3. Calculate churn percentage by payment method
payment_churn = pd.crosstab(
    dataset["PaymentMethod"],
    dataset["Churn"],
    normalize="index"
) * 100

# 4. Display results
print("\n========== CHURN BY PAYMENT METHOD ==========")
print(payment_churn)

# 5. Create visualization
payment_churn.plot(
    kind="bar",
    figsize=(12, 6),
    color=["seagreen", "tomato"]
)

# 6. Add chart labels
plt.title("Customer Churn by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Percentage of Customers")
plt.xticks(rotation=25, ha="right")
plt.legend(title="Churn")

# 7. Display chart
plt.tight_layout()
plt.show()

print("\nPAYMENT METHOD ANALYSIS COMPLETED!")
