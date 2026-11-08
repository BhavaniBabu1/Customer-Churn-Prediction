# ==========================================
# CUSTOMER CHURN PREDICTION PROJECT
# STEP 18 - STREAMING SERVICES ANALYSIS
# ==========================================

# 1. Import libraries
import pandas as pd
import matplotlib.pyplot as plt

# 2. Load dataset
dataset = pd.read_csv("dataset/telco_customer_churn.csv")

# 3. Analyze churn by Streaming TV
tv_churn = pd.crosstab(
    dataset["StreamingTV"],
    dataset["Churn"],
    normalize="index"
) * 100

# 4. Analyze churn by Streaming Movies
movies_churn = pd.crosstab(
    dataset["StreamingMovies"],
    dataset["Churn"],
    normalize="index"
) * 100

# 5. Display results
print("\n========== CHURN BY STREAMING TV ==========")
print(tv_churn.round(2))

print("\n========== CHURN BY STREAMING MOVIES ==========")
print(movies_churn.round(2))

# 6. Create two charts
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Streaming TV chart
tv_churn.plot(
    kind="bar",
    ax=axes[0],
    color=["seagreen", "tomato"]
)

axes[0].set_title("Customer Churn by Streaming TV")
axes[0].set_xlabel("Streaming TV")
axes[0].set_ylabel("Percentage of Customers")
axes[0].tick_params(axis="x", rotation=15)
axes[0].legend(title="Churn")

# Streaming Movies chart
movies_churn.plot(
    kind="bar",
    ax=axes[1],
    color=["seagreen", "tomato"]
)

axes[1].set_title("Customer Churn by Streaming Movies")
axes[1].set_xlabel("Streaming Movies")
axes[1].set_ylabel("Percentage of Customers")
axes[1].tick_params(axis="x", rotation=15)
axes[1].legend(title="Churn")

# 7. Adjust layout
plt.tight_layout()

# 8. Display charts
plt.show()

print("\nSTREAMING SERVICES ANALYSIS COMPLETED!")
