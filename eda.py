import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# ============================================================
# LOAN APPROVAL - EXPLORATORY DATA ANALYSIS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "loan_data.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATASET_PATH)

print("=" * 70)
print("LOAN APPROVAL - EXPLORATORY DATA ANALYSIS")
print("=" * 70)

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

print("\nFirst 5 Records:")
print(df.head())

print("\nLoan Status Distribution:")
print(df["Loan_Status"].value_counts())


# ============================================================
# 1. LOAN STATUS DISTRIBUTION
# ============================================================

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Loan_Status"
)

plt.title("Loan Approval Status Distribution")
plt.xlabel("Loan Status")
plt.ylabel("Number of Applicants")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "loan_status_distribution.png"
    )
)

plt.show()


# ============================================================
# 2. CREDIT HISTORY VS LOAN STATUS
# ============================================================

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Credit_History",
    hue="Loan_Status"
)

plt.title("Credit History vs Loan Approval")
plt.xlabel("Credit History")
plt.ylabel("Number of Applicants")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "credit_history_vs_loan_status.png"
    )
)

plt.show()


# ============================================================
# 3. EDUCATION VS LOAN STATUS
# ============================================================

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Education",
    hue="Loan_Status"
)

plt.title("Education vs Loan Approval")
plt.xlabel("Education")
plt.ylabel("Number of Applicants")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "education_vs_loan_status.png"
    )
)

plt.show()


# ============================================================
# 4. PROPERTY AREA VS LOAN STATUS
# ============================================================

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Property_Area",
    hue="Loan_Status"
)

plt.title("Property Area vs Loan Approval")
plt.xlabel("Property Area")
plt.ylabel("Number of Applicants")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "property_area_vs_loan_status.png"
    )
)

plt.show()


# ============================================================
# 5. APPLICANT INCOME DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="ApplicantIncome",
    bins=30,
    kde=True
)

plt.title("Applicant Income Distribution")
plt.xlabel("Applicant Income")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "applicant_income_distribution.png"
    )
)

plt.show()


# ============================================================
# 6. LOAN AMOUNT DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="LoanAmount",
    bins=30,
    kde=True
)

plt.title("Loan Amount Distribution")
plt.xlabel("Loan Amount")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "loan_amount_distribution.png"
    )
)

plt.show()


# ============================================================
# 7. INCOME VS LOAN AMOUNT
# ============================================================

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="ApplicantIncome",
    y="LoanAmount",
    hue="Loan_Status"
)

plt.title("Applicant Income vs Loan Amount")
plt.xlabel("Applicant Income")
plt.ylabel("Loan Amount")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "income_vs_loan_amount.png"
    )
)

plt.show()


# ============================================================
# 8. CORRELATION HEATMAP
# ============================================================

numeric_df = df.select_dtypes(include=["number"])

plt.figure(figsize=(10, 7))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Feature Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "correlation_heatmap.png"
    )
)

plt.show()


print("\n" + "=" * 70)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nGraphs saved in:")
print(OUTPUT_DIR)