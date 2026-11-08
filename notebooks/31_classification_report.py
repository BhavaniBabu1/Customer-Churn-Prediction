import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    classification_report,
    accuracy_score
)


# 1. LOAD DATASET
dataset = pd.read_csv("dataset/telco_customer_churn.csv")


# 2. CLEAN DATA
dataset["TotalCharges"] = pd.to_numeric(
    dataset["TotalCharges"], errors="coerce"
)

dataset["TotalCharges"] = dataset["TotalCharges"].fillna(0)


# 3. SEPARATE FEATURES AND TARGET
X = dataset.drop(columns=["customerID", "Churn"])
y = dataset["Churn"].map({"No": 0, "Yes": 1})


# 4. IDENTIFY FEATURE TYPES
numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()


# 5. SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 6. CREATE PREPROCESSOR
preprocessor = ColumnTransformer([
    ("numerical", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numerical_features),

    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_features)
])


# 7. CREATE MODEL
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])


# 8. TRAIN MODEL
model.fit(X_train, y_train)


# 9. MAKE PREDICTIONS
predictions = model.predict(X_test)


# 10. DISPLAY CLASSIFICATION REPORT
print("\n===== CLASSIFICATION REPORT =====")

print(classification_report(
    y_test,
    predictions,
    target_names=["Stayed", "Left"]
))


# 11. DISPLAY ACCURACY
accuracy = accuracy_score(y_test, predictions)

print("MODEL ACCURACY:", round(accuracy * 100, 2), "%")
