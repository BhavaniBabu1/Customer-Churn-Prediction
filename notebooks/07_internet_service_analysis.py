# ==========================================
# CUSTOMER CHURN PREDICTION PROJECT
# STEP 11 - INTERNET SERVICE ANALYSIS
# ==========================================

# 1. Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Load dataset
dataset = pd.read_csv("dataset/telco_customer_churn.csv")

# 3. Calculate churn percentage by internet service
internet_churn = pd.crosstab(
    dataset["InternetService"],
    dataset["Churn"],
    normalize="index"
) * 100

# 4. Display the result
print("\n========== CHURN BY INTERNET SERVICE ==========")
print(internet_churn)

# 5. Create a bar chart
internet_churn.plot(
    kind="bar",
    figsize=(10, 6),
    color=["seagreen", "tomato"]
)

# 6. Add chart details
plt.title("Customer Churn by Internet Service")
plt.xlabel("Internet Service Type")
plt.ylabel("Percentage of Customers")
plt.xticks(rotation=0)
plt.legend(title="Churn")

# 7. Display chart
plt.tight_layout()
plt.show()

print("\nINTERNET SERVICE ANALYSIS COMPLETED!")