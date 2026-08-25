import os
import joblib
import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)


MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_model.pkl"
)

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "scaler.pkl"
)

FEATURE_PATH = os.path.join(
    MODEL_DIR,
    "feature_names.pkl"
)


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(
    MODEL_PATH
)

scaler = joblib.load(
    SCALER_PATH
)

feature_names = joblib.load(
    FEATURE_PATH
)


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_loan(
    gender,
    married,
    dependents,
    education,
    self_employed,
    applicant_income,
    coapplicant_income,
    loan_amount,
    loan_term,
    credit_history,
    property_area
):

    # --------------------------------------------------------
    # Convert user input
    # --------------------------------------------------------

    gender_value = 1 if gender == "Male" else 0

    married_value = 1 if married == "Yes" else 0

    education_value = (
        1 if education == "Graduate"
        else 0
    )

    self_employed_value = (
        1 if self_employed == "Yes"
        else 0
    )

    if dependents == "3+":
        dependents_value = 3
    else:
        dependents_value = int(
            dependents
        )

    property_mapping = {
        "Urban": 2,
        "Semiurban": 1,
        "Rural": 0
    }

    property_value = property_mapping[
        property_area
    ]


    # --------------------------------------------------------
    # Create feature array
    # --------------------------------------------------------

    input_data = np.array([

        gender_value,
        married_value,
        dependents_value,
        education_value,
        self_employed_value,
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_term,
        credit_history,
        property_value

    ]).reshape(
        1,
        -1
    )


    # --------------------------------------------------------
    # Scale input
    # --------------------------------------------------------

    input_scaled = scaler.transform(
        input_data
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = model.predict(
        input_scaled
    )[0]


    # --------------------------------------------------------
    # Probability
    # --------------------------------------------------------

    probability = None

    if hasattr(
        model,
        "predict_proba"
    ):

        probabilities = model.predict_proba(
            input_scaled
        )

        probability = probabilities[
            0
        ][1]


    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    if prediction == 1:

        result = "LOAN APPROVED"

    else:

        result = "LOAN REJECTED"


    return result, probability


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("LOAN APPROVAL PREDICTION SYSTEM")
    print("=" * 70)


    gender = input(
        "\nGender (Male/Female): "
    )

    married = input(
        "Married (Yes/No): "
    )

    dependents = input(
        "Dependents (0/1/2/3+): "
    )

    education = input(
        "Education (Graduate/Not Graduate): "
    )

    self_employed = input(
        "Self Employed (Yes/No): "
    )

    applicant_income = float(
        input(
            "Applicant Income: "
        )
    )

    coapplicant_income = float(
        input(
            "Coapplicant Income: "
        )
    )

    loan_amount = float(
        input(
            "Loan Amount: "
        )
    )

    loan_term = float(
        input(
            "Loan Amount Term: "
        )
    )

    credit_history = float(
        input(
            "Credit History (1 = Good, 0 = Bad): "
        )
    )

    property_area = input(
        "Property Area (Urban/Semiurban/Rural): "
    )


    result, probability = predict_loan(

        gender,
        married,
        dependents,
        education,
        self_employed,
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_term,
        credit_history,
        property_area

    )


    print("\n" + "=" * 70)
    print("PREDICTION RESULT")
    print("=" * 70)

    print(
        "\nResult:",
        result
    )


    if probability is not None:

        print(
            "Approval Probability:",
            f"{probability * 100:.2f}%"
        )


    print("\n" + "=" * 70)