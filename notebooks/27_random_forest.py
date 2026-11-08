import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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

# Step 7: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Step 8: Numerical preprocessing
numerical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Step 9: Categorical preprocessing
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Step 10: Combine preprocessing
preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])

# Step 11: Create Random Forest model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    ))
])

# Step 12: Train model
model.fit(X_train, y_train)

# Step 13: Make predictions
y_pred = model.predict(X_test)

# Step 14: Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Step 15: Display results
print("\nRANDOM FOREST RESULTS")

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nActual values:")
print(y_test.head(10).tolist())

print("\nPredicted values:")
print(y_pred[:10].tolist())

print("\nRANDOM FOREST TRAINING COMPLETED!")
