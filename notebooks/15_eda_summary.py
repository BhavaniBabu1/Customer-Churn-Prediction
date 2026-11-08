import pandas as pd

# Load the dataset
dataset = pd.read_csv("dataset/telco_customer_churn.csv")

# Convert TotalCharges into numeric format
dataset["TotalCharges"] = pd.to_numeric(
    dataset["TotalCharges"], errors="coerce"
)

print("\n========== CUSTOMER CHURN EDA SUMMARY ==========")

# 1. Overall churn distribution
print("\n1. OVERALL CHURN")
print(dataset["Churn"].value_counts())

print("\nChurn Percentage:")
print(
    (dataset["Churn"].value_counts(normalize=True) * 100).round(2)
)

# 2. Contract analysis
print("\n2. CONTRACT TYPE VS CHURN (%)")
contract_analysis = pd.crosstab(
    dataset["Contract"],
    dataset["Churn"],
    normalize="index"
) * 100

print(contract_analysis.round(2))

# 3. Internet service analysis
print("\n3. INTERNET SERVICE VS CHURN (%)")
internet_analysis = pd.crosstab(
    dataset["InternetService"],
    dataset["Churn"],
    normalize="index"
) * 100

print(internet_analysis.round(2))

# 4. Tenure analysis
print("\n4. TENURE VS CHURN (%)")
dataset["TenureGroup"] = pd.cut(
    dataset["tenure"],
    bins=[-1, 12, 24, 48, float("inf")],
    labels=["0-12 Months", "13-24 Months", "25-48 Months", "49+ Months"]
)

tenure_analysis = pd.crosstab(
    dataset["TenureGroup"],
    dataset["Churn"],
    normalize="index"
) * 100

print(tenure_analysis.round(2))

# 5. Monthly charges analysis
print("\n5. MONTHLY CHARGES VS CHURN (%)")
dataset["ChargeGroup"] = pd.cut(
    dataset["MonthlyCharges"],
    bins=[0, 35, 70, float("inf")],
    labels=["Low", "Medium", "High"],
    include_lowest=True
)

charges_analysis = pd.crosstab(
    dataset["ChargeGroup"],
    dataset["Churn"],
    normalize="index"
) * 100

print(charges_analysis.round(2))

print("\n========== EDA SUMMARY COMPLETED ==========")
