import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Step 1: Load dataset
dataset = pd.read_csv("dataset/telco_customer_churn.csv")

# Step 2: Convert TotalCharges into numeric format
dataset["TotalCharges"] = pd.to_numeric(
    dataset["TotalCharges"],
    errors="coerce"
)

# Step 3: Remove customerID
dataset = dataset.drop(columns=["customerID"])

# Step 4: Separate features and target
X = dataset.drop(columns=["Churn"])
y = dataset["Churn"].map({"No": 0, "Yes": 1})

# Step 5: Identify numerical features
numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

# Step 6: Identify categorical features
categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

# Step 7: Create numerical preprocessing pipeline
numerical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Step 8: Create categorical preprocessing pipeline
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Step 9: Combine both pipelines
preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])

# Step 10: Transform the features
X_processed = preprocessor.fit_transform(X)

# Step 11: Display results
print("\nORIGINAL FEATURES SHAPE:", X.shape)
print("PROCESSED FEATURES SHAPE:", X_processed.shape)

print("\nTARGET SHAPE:", y.shape)
print("MISSING TARGET VALUES:", y.isnull().sum())

print("\nPREPROCESSING PIPELINE CREATED SUCCESSFULLY!")
