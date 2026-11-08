import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
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


# 7. CREATE LOGISTIC REGRESSION PIPELINE
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])


# 8. TRAIN MODEL
model.fit(X_train, y_train)


# 9. MAKE PREDICTIONS
predictions = model.predict(X_test)


# 10. GENERATE CONFUSION MATRIX
cm = confusion_matrix(y_test, predictions)

print("\nCONFUSION MATRIX")
print(cm)


# 11. DISPLAY CONFUSION MATRIX
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Stayed (No)", "Left (Yes)"]
)

display.plot(cmap="Blues", values_format="d")

plt.title("Logistic Regression - Confusion Matrix")
plt.tight_layout()
plt.show()


# 12. CLASSIFICATION REPORT
print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_test,
    predictions,
    target_names=["Stayed", "Left"]
))
