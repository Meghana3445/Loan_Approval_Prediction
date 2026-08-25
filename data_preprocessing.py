import pandas as pd
import os


# ============================================================
# LOAN APPROVAL DATA PREPROCESSING
# ============================================================

# Get project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset path
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "loan_data.csv")


def load_data():
    """Load the loan dataset."""

    if not os.path.exists(DATASET_PATH):
        print("ERROR: Dataset file not found!")
        print("Expected location:")
        print(DATASET_PATH)
        return None

    df = pd.read_csv(DATASET_PATH)

    print("=" * 70)
    print("LOAN APPROVAL DATASET")
    print("=" * 70)

    print("\nDataset loaded successfully!")

    print("\nDataset Shape:")
    print(df.shape)

    print("\nColumns:")
    print(list(df.columns))

    print("\nFirst 5 Records:")
    print(df.head())

    print("\nMissing Values:")
    print(df.isnull().sum())

    return df


def preprocess_data(df):
    """Clean and preprocess the dataset."""

    print("\n" + "=" * 70)
    print("DATA PREPROCESSING")
    print("=" * 70)

    # Remove Loan_ID because it is only an identifier
    if "Loan_ID" in df.columns:
        df = df.drop("Loan_ID", axis=1)

    # --------------------------------------------------------
    # Handle missing values
    # --------------------------------------------------------

    categorical_columns = [
        "Gender",
        "Married",
        "Dependents",
        "Self_Employed",
        "Credit_History",
        "Property_Area",
        "Education"
    ]

    numerical_columns = [
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term"
    ]

    for column in categorical_columns:
        if column in df.columns:
            df[column] = df[column].fillna(df[column].mode()[0])

    for column in numerical_columns:
        if column in df.columns:
            df[column] = df[column].fillna(df[column].median())

    # --------------------------------------------------------
    # Convert categorical variables
    # --------------------------------------------------------

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

    # Property Area
    if "Property_Area" in df.columns:
        property_mapping = {
            "Urban": 2,
            "Semiurban": 1,
            "Rural": 0
        }

        df["Property_Area"] = df["Property_Area"].map(
            property_mapping
        )

    # Dependents
    if "Dependents" in df.columns:
        df["Dependents"] = df["Dependents"].replace({
            "3+": 3
        })

        df["Dependents"] = pd.to_numeric(
            df["Dependents"],
            errors="coerce"
        )

        df["Dependents"] = df["Dependents"].fillna(
            df["Dependents"].median()
        )

    # Credit History
    if "Credit_History" in df.columns:
        df["Credit_History"] = pd.to_numeric(
            df["Credit_History"],
            errors="coerce"
        )

        df["Credit_History"] = df["Credit_History"].fillna(
            df["Credit_History"].median()
        )

    # --------------------------------------------------------
    # Convert target variable
    # --------------------------------------------------------

    if "Loan_Status" in df.columns:

        df["Loan_Status"] = df["Loan_Status"].map({
            "Y": 1,
            "N": 0,
            "Approved": 1,
            "Rejected": 0
        })

        df["Loan_Status"] = pd.to_numeric(
            df["Loan_Status"],
            errors="coerce"
        )

    # Remove rows where target is missing
    if "Loan_Status" in df.columns:
        df = df.dropna(subset=["Loan_Status"])

    # Final missing-value check
    df = df.fillna(0)

    print("\nPreprocessing completed successfully!")

    print("\nProcessed Dataset Shape:")
    print(df.shape)

    print("\nProcessed Columns:")
    print(list(df.columns))

    print("\nRemaining Missing Values:")
    print(df.isnull().sum())

    return df


if __name__ == "__main__":

    data = load_data()

    if data is not None:

        processed_data = preprocess_data(data)

        print("\n" + "=" * 70)
        print("PREPROCESSED DATA")
        print("=" * 70)

        print(processed_data.head())

        print("\nLoan Status Distribution:")
        print(processed_data["Loan_Status"].value_counts())

        print("\n" + "=" * 70)
        print("PREPROCESSING COMPLETED SUCCESSFULLY")
        print("=" * 70)