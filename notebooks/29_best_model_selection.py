import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
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


# 6. PREPROCESSING FUNCTION
def create_preprocessor():

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("numerical", numerical_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])


# 7. DEFINE MODELS
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
}


# 8. TRAIN AND COMPARE
results = []
trained_models = {}

for model_name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("classifier", model)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    trained_models[model_name] = pipeline

    results.append({
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1 Score": f1_score(y_test, predictions, zero_division=0)
    })


# 9. CREATE RESULTS TABLE
results_df = pd.DataFrame(results)

print("\n===== MODEL PERFORMANCE =====")
print(results_df.round(4).to_string(index=False))


# 10. SELECT BEST MODEL USING F1 SCORE
best_model_name = results_df.loc[
    results_df["F1 Score"].idxmax(),
    "Model"
]

best_model = trained_models[best_model_name]

print("\n===== BEST MODEL SELECTED =====")
print("Best Model:", best_model_name)

print(
    "Best F1 Score:",
    round(results_df["F1 Score"].max(), 4)
)


# 11. VISUALIZE MODEL PERFORMANCE
results_df.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1 Score"]
].plot(kind="bar", figsize=(11, 6))

plt.title("Machine Learning Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()
