import pandas as pd
import numpy as np
import os
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "loan_data.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("LOAN APPROVAL PREDICTION - MODEL TRAINING")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully!")

print("\nDataset Shape:")
print(df.shape)


# ============================================================
# DATA PREPROCESSING
# ============================================================

print("\n" + "=" * 70)
print("DATA PREPROCESSING")
print("=" * 70)


# Remove Loan_ID
if "Loan_ID" in df.columns:
    df = df.drop("Loan_ID", axis=1)


# ------------------------------------------------------------
# Fill missing categorical values
# ------------------------------------------------------------

categorical_columns = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area"
]

for column in categorical_columns:

    if column in df.columns:

        df[column] = df[column].fillna(
            df[column].mode()[0]
        )


# ------------------------------------------------------------
# Fill missing numerical values
# ------------------------------------------------------------

numerical_columns = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History"
]

for column in numerical_columns:

    if column in df.columns:

        df[column] = df[column].fillna(
            df[column].median()
        )


# ============================================================
# ENCODING CATEGORICAL VARIABLES
# ============================================================

# Gender
if "Gender" in df.columns:

    df["Gender"] = df["Gender"].map({
        "Male": 1,
        "Female": 0
    })


# Married
if "Married" in df.columns:

    df["Married"] = df["Married"].map({
        "Yes": 1,
        "No": 0
    })


# Education
if "Education" in df.columns:

    df["Education"] = df["Education"].map({
        "Graduate": 1,
        "Not Graduate": 0
    })


# Self Employed
if "Self_Employed" in df.columns:

    df["Self_Employed"] = df["Self_Employed"].map({
        "Yes": 1,
        "No": 0
    })


# Dependents
if "Dependents" in df.columns:

    df["Dependents"] = df["Dependents"].replace({
        "3+": 3
    })

    df["Dependents"] = pd.to_numeric(
        df["Dependents"],
        errors="coerce"
    )

    df["Dependents"] = df["Dependents"].fillna(0)


# Property Area
if "Property_Area" in df.columns:

    df["Property_Area"] = df["Property_Area"].map({
        "Urban": 2,
        "Semiurban": 1,
        "Rural": 0
    })


# Credit History
if "Credit_History" in df.columns:

    df["Credit_History"] = pd.to_numeric(
        df["Credit_History"],
        errors="coerce"
    )

    df["Credit_History"] = df["Credit_History"].fillna(0)


# ============================================================
# TARGET VARIABLE
# ============================================================

df["Loan_Status"] = df["Loan_Status"].map({
    "Y": 1,
    "N": 0,
    "Approved": 1,
    "Rejected": 0
})


# Remove rows with invalid target
df = df.dropna(
    subset=["Loan_Status"]
)


# Fill remaining missing values
df = df.fillna(0)


# ============================================================
# FEATURES AND TARGET
# ============================================================

X = df.drop(
    "Loan_Status",
    axis=1
)

y = df["Loan_Status"]


print("\nFeatures:")
print(list(X.columns))

print("\nNumber of Features:")
print(X.shape[1])

print("\nTarget Distribution:")
print(y.value_counts())


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n" + "=" * 70)
print("TRAIN TEST SPLIT")
print("=" * 70)

print("\nTraining Records:", len(X_train))
print("Testing Records :", len(X_test))


# ============================================================
# FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# Save scaler
joblib.dump(
    scaler,
    os.path.join(
        MODEL_DIR,
        "scaler.pkl"
    )
)


# ============================================================
# DEFINE MODELS
# ============================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ),

    "Naive Bayes":
        GaussianNB(),

    "SVM":
        SVC(
            kernel="rbf",
            probability=True,
            random_state=42
        )
}


# ============================================================
# TRAIN MODELS
# ============================================================

results = []

trained_models = {}

print("\n" + "=" * 70)
print("MODEL TRAINING")
print("=" * 70)


for name, model in models.items():

    print("\nTraining:", name)

    # Use scaled data
    model.fit(
        X_train_scaled,
        y_train
    )

    # Prediction
    y_pred = model.predict(
        X_test_scaled
    )

    # Metrics
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1-Score  : {f1:.4f}"
    )

    results.append({

        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1

    })

    trained_models[name] = model


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# FIND BEST MODEL
# ============================================================

best_index = results_df[
    "Accuracy"
].idxmax()

best_model_name = results_df.loc[
    best_index,
    "Model"
]

best_accuracy = results_df.loc[
    best_index,
    "Accuracy"
]

best_model = trained_models[
    best_model_name
]


print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print(
    "\nBest Model:",
    best_model_name
)

print(
    "Accuracy :",
    f"{best_accuracy:.4f}"
)


# ============================================================
# SAVE BEST MODEL
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "best_model.pkl"
)

joblib.dump(
    best_model,
    model_path
)


# Save feature names
feature_path = os.path.join(
    MODEL_DIR,
    "feature_names.pkl"
)

joblib.dump(
    list(X.columns),
    feature_path
)


print("\nBest model saved to:")
print(model_path)


# ============================================================
# MODEL COMPARISON GRAPH
# ============================================================

plt.figure(
    figsize=(10, 6)
)

sns.barplot(
    data=results_df,
    x="Model",
    y="Accuracy"
)

plt.title(
    "Loan Approval Model Accuracy Comparison"
)

plt.xlabel("Machine Learning Model")
plt.ylabel("Accuracy")

plt.xticks(
    rotation=20
)

plt.ylim(
    0,
    1
)

plt.tight_layout()

comparison_path = os.path.join(
    OUTPUT_DIR,
    "model_comparison.png"
)

plt.savefig(
    comparison_path
)

plt.show()


# ============================================================
# CONFUSION MATRIX
# ============================================================

y_pred_best = best_model.predict(
    X_test_scaled
)

cm = confusion_matrix(
    y_test,
    y_pred_best
)


plt.figure(
    figsize=(7, 5)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "Rejected",
        "Approved"
    ],
    yticklabels=[
        "Rejected",
        "Approved"
    ]
)

plt.title(
    f"Confusion Matrix - {best_model_name}"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.tight_layout()

confusion_path = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.png"
)

plt.savefig(
    confusion_path
)

plt.show()


# ============================================================
# SAVE RESULTS
# ============================================================

results_path = os.path.join(
    OUTPUT_DIR,
    "model_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


# ============================================================
# COMPLETED
# ============================================================

print("\n" + "=" * 70)
print("MODEL TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nGenerated files:")

print(
    "\nBest Model:"
)

print(
    "models/best_model.pkl"
)

print(
    "\nScaler:"
)

print(
    "models/scaler.pkl"
)

print(
    "\nFeature Names:"
)

print(
    "models/feature_names.pkl"
)

print(
    "\nModel Comparison:"
)

print(
    "outputs/model_comparison.png"
)

print(
    "\nConfusion Matrix:"
)

print(
    "outputs/confusion_matrix.png"
)

print(
    "\nResults:"
)

print(
    "outputs/model_results.csv"
)

print("\n" + "=" * 70)