import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    roc_curve,
    roc_auc_score
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


# 9. GET PREDICTION PROBABILITIES
probabilities = model.predict_proba(X_test)[:, 1]


# 10. CALCULATE AUC SCORE
auc_score = roc_auc_score(y_test, probabilities)

print("\n===== ROC AUC ANALYSIS =====")
print("AUC Score:", round(auc_score, 4))


# 11. CALCULATE ROC CURVE
fpr, tpr, thresholds = roc_curve(
    y_test,
    probabilities
)


# 12. VISUALIZE ROC CURVE
plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"Logistic Regression (AUC = {auc_score:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Customer Churn Prediction")

plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
