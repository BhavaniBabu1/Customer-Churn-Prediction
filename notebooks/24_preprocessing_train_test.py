import pandas as pd

from sklearn.model_selection import train_test_split
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

# Step 3: Remove unnecessary customerID column
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

# Step 7: Split dataset before preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Step 8: Create numerical pipeline
numerical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Step 9: Create categorical pipeline
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Step 10: Combine preprocessing pipelines
preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])

# Step 11: Fit and transform training data
X_train_processed = preprocessor.fit_transform(X_train)

# Step 12: Transform testing data
X_test_processed = preprocessor.transform(X_test)

# Step 13: Display results
print("\nTRAINING DATA BEFORE PROCESSING:", X_train.shape)
print("TESTING DATA BEFORE PROCESSING:", X_test.shape)

print("\nTRAINING DATA AFTER PROCESSING:", X_train_processed.shape)
print("TESTING DATA AFTER PROCESSING:", X_test_processed.shape)

print("\nTRAINING TARGET:", y_train.shape)
print("TESTING TARGET:", y_test.shape)

print("\nMISSING VALUES IN PROCESSED TRAINING DATA:",
      X_train_processed.shape[0] == X_train.shape[0])

print("\nPREPROCESSING COMPLETED SUCCESSFULLY!")
