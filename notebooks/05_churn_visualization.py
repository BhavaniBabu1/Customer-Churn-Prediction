# ==========================================
# CUSTOMER CHURN PREDICTION PROJECT
# STEP 9 - CHURN VISUALIZATION
# ==========================================

# 1. Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Load dataset
dataset = pd.read_csv("dataset/telco_customer_churn.csv")

# 3. Set visualization style
sns.set_style("whitegrid")

# 4. Create a figure with two plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 5. Count plot
sns.countplot(
    data=dataset,
    x="Churn",
    hue="Churn",
    palette="Set2",
    legend=False,
    ax=axes[0]
)

axes[0].set_title("Customer Churn Count")
axes[0].set_xlabel("Churn Status")
axes[0].set_ylabel("Number of Customers")

# 6. Pie chart
churn_counts = dataset["Churn"].value_counts()

axes[1].pie(
    churn_counts,
    labels=churn_counts.index,
    autopct="%1.1f%%",
    startangle=90
)

axes[1].set_title("Customer Churn Percentage")

# 7. Adjust layout
plt.tight_layout()

# 8. Display visualization
plt.show()
