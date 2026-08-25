import tkinter as tk
from tkinter import ttk, messagebox
import os
import joblib
import numpy as np


# ============================================================
# PATH CONFIGURATION
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


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

try:

    model = joblib.load(MODEL_PATH)

    scaler = joblib.load(SCALER_PATH)

except Exception as e:

    messagebox.showerror(
        "Model Error",
        f"Unable to load trained model.\n\n{e}"
    )

    raise SystemExit


# ============================================================
# MAIN WINDOW
# ============================================================

root = tk.Tk()

root.title(
    "Loan Approval Prediction System"
)

root.geometry(
    "900x720"
)

root.resizable(
    False,
    False
)


# ============================================================
# STYLE
# ============================================================

style = ttk.Style()

try:
    style.theme_use("clam")
except:
    pass


style.configure(
    "Title.TLabel",
    font=("Arial", 22, "bold")
)

style.configure(
    "Subtitle.TLabel",
    font=("Arial", 11)
)

style.configure(
    "Section.TLabel",
    font=("Arial", 13, "bold")
)

style.configure(
    "TButton",
    font=("Arial", 10, "bold"),
    padding=8
)


# ============================================================
# HEADER
# ============================================================

header = ttk.Frame(
    root,
    padding=15
)

header.pack(
    fill="x"
)


title = ttk.Label(
    header,
    text="LOAN APPROVAL PREDICTION SYSTEM",
    style="Title.TLabel"
)

title.pack()


subtitle = ttk.Label(
    header,
    text="Machine Learning Based Loan Eligibility Prediction",
    style="Subtitle.TLabel"
)

subtitle.pack(
    pady=5
)


# ============================================================
# MAIN CONTAINER
# ============================================================

container = ttk.Frame(
    root,
    padding=15
)

container.pack(
    fill="both",
    expand=True
)


# ============================================================
# LEFT FRAME
# ============================================================

left_frame = ttk.LabelFrame(
    container,
    text="Applicant Information",
    padding=15
)

left_frame.grid(
    row=0,
    column=0,
    padx=10,
    pady=10,
    sticky="n"
)


# ============================================================
# RIGHT FRAME
# ============================================================

right_frame = ttk.LabelFrame(
    container,
    text="Financial & Loan Information",
    padding=15
)

right_frame.grid(
    row=0,
    column=1,
    padx=10,
    pady=10,
    sticky="n"
)


# ============================================================
# VARIABLES
# ============================================================

gender_var = tk.StringVar()
married_var = tk.StringVar()
dependents_var = tk.StringVar()
education_var = tk.StringVar()
self_employed_var = tk.StringVar()

credit_history_var = tk.StringVar()
property_area_var = tk.StringVar()


# ============================================================
# HELPER FUNCTION
# ============================================================

def create_label(
    parent,
    text,
    row
):

    label = ttk.Label(
        parent,
        text=text
    )

    label.grid(
        row=row,
        column=0,
        padx=10,
        pady=10,
        sticky="w"
    )

    return label


def create_combo(
    parent,
    variable,
    values,
    row
):

    combo = ttk.Combobox(
        parent,
        textvariable=variable,
        values=values,
        state="readonly",
        width=25
    )

    combo.grid(
        row=row,
        column=1,
        padx=10,
        pady=10
    )

    return combo


# ============================================================
# APPLICANT INFORMATION
# ============================================================

create_label(
    left_frame,
    "Gender:",
    0
)

create_combo(
    left_frame,
    gender_var,
    ["Male", "Female"],
    0
)


create_label(
    left_frame,
    "Married:",
    1
)

create_combo(
    left_frame,
    married_var,
    ["Yes", "No"],
    1
)


create_label(
    left_frame,
    "Dependents:",
    2
)

create_combo(
    left_frame,
    dependents_var,
    ["0", "1", "2", "3+"],
    2
)


create_label(
    left_frame,
    "Education:",
    3
)

create_combo(
    left_frame,
    education_var,
    ["Graduate", "Not Graduate"],
    3
)


create_label(
    left_frame,
    "Self Employed:",
    4
)

create_combo(
    left_frame,
    self_employed_var,
    ["Yes", "No"],
    4
)


# ============================================================
# ENTRY VARIABLES
# ============================================================

applicant_income_entry = ttk.Entry(
    right_frame,
    width=28
)

coapplicant_income_entry = ttk.Entry(
    right_frame,
    width=28
)

loan_amount_entry = ttk.Entry(
    right_frame,
    width=28
)

loan_term_entry = ttk.Entry(
    right_frame,
    width=28
)


# ============================================================
# FINANCIAL INFORMATION
# ============================================================

create_label(
    right_frame,
    "Applicant Income:",
    0
)

applicant_income_entry.grid(
    row=0,
    column=1,
    padx=10,
    pady=10
)


create_label(
    right_frame,
    "Coapplicant Income:",
    1
)

coapplicant_income_entry.grid(
    row=1,
    column=1,
    padx=10,
    pady=10
)


create_label(
    right_frame,
    "Loan Amount:",
    2
)

loan_amount_entry.grid(
    row=2,
    column=1,
    padx=10,
    pady=10
)


create_label(
    right_frame,
    "Loan Term:",
    3
)

loan_term_entry.grid(
    row=3,
    column=1,
    padx=10,
    pady=10
)


# ============================================================
# CREDIT HISTORY
# ============================================================

create_label(
    right_frame,
    "Credit History:",
    4
)

create_combo(
    right_frame,
    credit_history_var,
    ["1 - Good", "0 - Bad"],
    4
)


# ============================================================
# PROPERTY AREA
# ============================================================

create_label(
    right_frame,
    "Property Area:",
    5
)

create_combo(
    right_frame,
    property_area_var,
    [
        "Urban",
        "Semiurban",
        "Rural"
    ],
    5
)


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_loan():

    try:

        # ----------------------------------------------------
        # Validate dropdowns
        # ----------------------------------------------------

        if not gender_var.get():
            messagebox.showwarning(
                "Missing Input",
                "Please select Gender."
            )
            return

        if not married_var.get():
            messagebox.showwarning(
                "Missing Input",
                "Please select Married status."
            )
            return

        if not dependents_var.get():
            messagebox.showwarning(
                "Missing Input",
                "Please select Dependents."
            )
            return

        if not education_var.get():
            messagebox.showwarning(
                "Missing Input",
                "Please select Education."
            )
            return

        if not self_employed_var.get():
            messagebox.showwarning(
                "Missing Input",
                "Please select Self Employed status."
            )
            return

        if not credit_history_var.get():
            messagebox.showwarning(
                "Missing Input",
                "Please select Credit History."
            )
            return

        if not property_area_var.get():
            messagebox.showwarning(
                "Missing Input",
                "Please select Property Area."
            )
            return


        # ----------------------------------------------------
        # Numeric validation
        # ----------------------------------------------------

        applicant_income = float(
            applicant_income_entry.get()
        )

        coapplicant_income = float(
            coapplicant_income_entry.get()
        )

        loan_amount = float(
            loan_amount_entry.get()
        )

        loan_term = float(
            loan_term_entry.get()
        )


        if applicant_income < 0:
            raise ValueError(
                "Applicant income cannot be negative."
            )

        if coapplicant_income < 0:
            raise ValueError(
                "Coapplicant income cannot be negative."
            )

        if loan_amount <= 0:
            raise ValueError(
                "Loan amount must be greater than zero."
            )

        if loan_term <= 0:
            raise ValueError(
                "Loan term must be greater than zero."
            )


        # ----------------------------------------------------
        # Encode values
        # ----------------------------------------------------

        gender = (
            1
            if gender_var.get() == "Male"
            else 0
        )

        married = (
            1
            if married_var.get() == "Yes"
            else 0
        )

        dependents_text = dependents_var.get()

        if dependents_text == "3+":

            dependents = 3

        else:

            dependents = int(
                dependents_text
            )


        education = (
            1
            if education_var.get() == "Graduate"
            else 0
        )

        self_employed = (
            1
            if self_employed_var.get() == "Yes"
            else 0
        )


        if credit_history_var.get().startswith("1"):

            credit_history = 1

        else:

            credit_history = 0


        property_mapping = {
            "Urban": 2,
            "Semiurban": 1,
            "Rural": 0
        }

        property_area = property_mapping[
            property_area_var.get()
        ]


        # ----------------------------------------------------
        # Create feature vector
        # ----------------------------------------------------

        input_data = np.array([

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

        ]).reshape(
            1,
            -1
        )


        # ----------------------------------------------------
        # Scale
        # ----------------------------------------------------

        input_scaled = scaler.transform(
            input_data
        )


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = model.predict(
            input_scaled
        )[0]


        # ----------------------------------------------------
        # Probability
        # ----------------------------------------------------

        probability = None

        if hasattr(
            model,
            "predict_proba"
        ):

            probability = (
                model.predict_proba(
                    input_scaled
                )[0][1] * 100
            )


        # ----------------------------------------------------
        # Display result
        # ----------------------------------------------------

        if prediction == 1:

            result_label.config(
                text="✓ LOAN APPROVED",
                foreground="green"
            )

        else:

            result_label.config(
                text="✗ LOAN REJECTED",
                foreground="red"
            )


        if probability is not None:

            probability_label.config(
                text=(
                    f"Approval Probability: "
                    f"{probability:.2f}%"
                )
            )

        else:

            probability_label.config(
                text=""
            )


    except ValueError as e:

        messagebox.showerror(
            "Invalid Input",
            str(e)
        )

    except Exception as e:

        messagebox.showerror(
            "Prediction Error",
            str(e)
        )


# ============================================================
# CLEAR FUNCTION
# ============================================================

def clear_fields():

    gender_var.set("")
    married_var.set("")
    dependents_var.set("")
    education_var.set("")
    self_employed_var.set("")
    credit_history_var.set("")
    property_area_var.set("")

    applicant_income_entry.delete(
        0,
        tk.END
    )

    coapplicant_income_entry.delete(
        0,
        tk.END
    )

    loan_amount_entry.delete(
        0,
        tk.END
    )

    loan_term_entry.delete(
        0,
        tk.END
    )

    result_label.config(
        text=""
    )

    probability_label.config(
        text=""
    )


# ============================================================
# BUTTON FRAME
# ============================================================

button_frame = ttk.Frame(
    root,
    padding=10
)

button_frame.pack()


predict_button = ttk.Button(
    button_frame,
    text="PREDICT LOAN APPROVAL",
    command=predict_loan
)

predict_button.grid(
    row=0,
    column=0,
    padx=10
)


clear_button = ttk.Button(
    button_frame,
    text="CLEAR",
    command=clear_fields
)

clear_button.grid(
    row=0,
    column=1,
    padx=10
)


# ============================================================
# RESULT FRAME
# ============================================================

result_frame = ttk.LabelFrame(
    root,
    text="Prediction Result",
    padding=15
)

result_frame.pack(
    fill="x",
    padx=30,
    pady=10
)


result_label = tk.Label(
    result_frame,
    text="",
    font=("Arial", 20, "bold")
)

result_label.pack(
    pady=5
)


probability_label = tk.Label(
    result_frame,
    text="",
    font=("Arial", 12)
)

probability_label.pack(
    pady=5
)


# ============================================================
# INFORMATION
# ============================================================

info_label = tk.Label(
    root,
    text=(
        "This system uses machine learning to predict "
        "loan approval based on applicant information."
    ),
    font=("Arial", 9)
)

info_label.pack(
    pady=5
)


footer_label = tk.Label(
    root,
    text="For educational purposes only",
    font=("Arial", 9)
)

footer_label.pack(
    pady=5
)


# ============================================================
# START GUI
# ============================================================

root.mainloop()