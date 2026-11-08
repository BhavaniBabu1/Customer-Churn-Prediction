# Telco Customer Churn Prediction

A Machine Learning-based application that predicts whether a telecommunications customer is likely to leave a service provider. Built using Python, Scikit-learn, and Streamlit.

## Project Overview

Customer churn occurs when customers stop using a company's services. Predicting churn helps businesses identify customers who may leave and take proactive steps to improve customer retention.

This project uses customer information from the IBM Telco Customer Churn dataset to train machine learning models and predict customer churn.

## Features

- Exploratory Data Analysis (EDA)
- Data cleaning and preprocessing
- Categorical feature encoding
- Feature scaling
- Train-test splitting
- Machine learning model training and comparison
- Model evaluation using classification metrics
- Churn probability prediction
- Interactive Streamlit web application

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Streamlit
- Joblib

## Dataset

**Source:** IBM Telco Customer Churn dataset

- 7,043 customer records
- 21 original columns
- Target variable: `Churn`
- Target classes: Yes and No

## Machine Learning Models

The following algorithms were evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest

A class-weight-balanced Logistic Regression model was selected for the saved application model, prioritizing churn detection.

## Model Evaluation

| Metric | Balanced Logistic Regression |
|---|---:|
| Accuracy | 73.81% |
| Precision | 50.43% |
| Recall | 78.34% |
| F1 Score | 61.36% |
| ROC-AUC | 0.8416 |

The model uses a probability threshold of 0.5.

## Project Structure

```text
Customer-Churn-Prediction/
│
├── dataset/
│   └── telco_customer_churn.csv
│
├── models/
│   └── churn_model.joblib
│
├── notebooks/
│   └── Data analysis and ML scripts
│
├── app.py
├── requirements.txt
├── test_libraries.py
├── .gitignore
└── README.md
```

## Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/BhavaniBabu1/Customer-Churn-Prediction.git
```

### 2. Navigate to the project folder

```bash
cd Customer-Churn-Prediction
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal to use the application.

## Future Improvements

- Hyperparameter tuning
- Additional machine learning algorithms
- Improved customer retention recommendations
- Model explainability using SHAP
- Deployment to a cloud hosting platform

## Author

**Bhavani B**

B.Tech – Artificial Intelligence and Data Science

GitHub: [BhavaniBabu1](https://github.com/BhavaniBabu1)

---

*This project was developed as a hands-on application of machine learning, data analysis, and web application development.*
