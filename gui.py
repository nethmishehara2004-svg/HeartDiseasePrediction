
import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import numpy as np
from tensorflow.keras.models import load_model


# Load the existing trained model and scaler
try:
    model = load_model("heart_model.keras")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    raise RuntimeError(
        "Could not load heart_model.keras or scaler.pkl. "
        "Make sure both files are in the project folder."
    ) from e


# Main application window
root = tk.Tk()
root.title("Heart Disease Prediction System")
root.geometry("650x750")
root.resizable(False, False)

# Theme colors
BG_COLOR = "#F3F7FA"
HEADER_COLOR = "#123B5D"
ACCENT_COLOR = "#008C95"
TEXT_COLOR = "#203040"

root.configure(bg=BG_COLOR)

# Title
title_label = tk.Label(
    root,
    text="Heart Disease Prediction System",
    font=("Arial", 19, "bold"),
    bg=HEADER_COLOR,
    fg="white",
    pady=18
)
title_label.pack(fill="x")

subtitle = tk.Label(
    root,
    text="Enter patient information and clinical measurements",
    font=("Arial", 10),
    bg=BG_COLOR,
    fg=TEXT_COLOR,
    pady=10
)
subtitle.pack()


# Main form
form = tk.Frame(root, bg=BG_COLOR, padx=35, pady=10)
form.pack(fill="both", expand=True)

fields = {}


def add_dropdown(label, options):
    """Add a labelled dropdown to the form."""
    row = len(fields)

    label_widget = tk.Label(
        form,
        text=label,
        font=("Arial", 10),
        bg=BG_COLOR,
        fg=TEXT_COLOR,
        anchor="w"
    )
    label_widget.grid(
        row=row,
        column=0,
        sticky="w",
        padx=(0, 20),
        pady=5
    )

    variable = tk.StringVar(value=options[0])

    dropdown = ttk.Combobox(
        form,
        textvariable=variable,
        values=options,
        state="readonly",
        width=27
    )
    dropdown.grid(
        row=row,
        column=1,
        sticky="ew",
        pady=5
    )

    fields[label] = variable


def add_entry(label, default_value):
    """Add a labelled numeric input."""
    row = len(fields)

    label_widget = tk.Label(
        form,
        text=label,
        font=("Arial", 10),
        bg=BG_COLOR,
        fg=TEXT_COLOR,
        anchor="w"
    )
    label_widget.grid(
        row=row,
        column=0,
        sticky="w",
        padx=(0, 20),
        pady=5
    )

    variable = tk.StringVar(value=str(default_value))

    entry = ttk.Entry(
        form,
        textvariable=variable,
        width=30
    )
    entry.grid(
        row=row,
        column=1,
        sticky="ew",
        pady=5
    )

    fields[label] = variable


# Patient information
add_entry("Age (years)", 50)
add_dropdown("Gender", ["Male", "Female"])

# Clinical measurements
add_dropdown(
    "Chest Pain Type",
    [
        "Typical Angina",
        "Atypical Angina",
        "Non-anginal Pain",
        "Asymptomatic"
    ]
)

add_entry("Resting Blood Pressure (mmHg)", 120)
add_entry("Cholesterol (mg/dL)", 200)

add_dropdown(
    "Fasting Blood Sugar > 120 mg/dL",
    ["No", "Yes"]
)

add_dropdown(
    "Resting ECG",
    [
        "Normal",
        "ST-T Abnormality",
        "Left Ventricular Hypertrophy"
    ]
)

add_entry("Maximum Heart Rate", 150)

add_dropdown(
    "Exercise-Induced Angina",
    ["No", "Yes"]
)

add_entry("ST Depression (Oldpeak)", 1.0)

add_dropdown(
    "ST Segment Slope",
    [
        "Upsloping",
        "Flat",
        "Downsloping"
    ]
)

add_dropdown(
    "Number of Major Vessels (CA)",
    ["0", "1", "2", "3"]
)

add_dropdown(
    "Thalassemia",
    [
        "Normal",
        "Fixed Defect",
        "Reversible Defect"
    ]
)

form.columnconfigure(1, weight=1)


def predict():
    """Validate inputs and predict using the existing ANN model."""
    try:
        age = float(fields["Age (years)"].get())
        trestbps = float(
            fields["Resting Blood Pressure (mmHg)"].get()
        )
        chol = float(fields["Cholesterol (mg/dL)"].get())
        thalach = float(fields["Maximum Heart Rate"].get())
        oldpeak = float(fields["ST Depression (Oldpeak)"].get())

        # Basic input validation
        if not 1 <= age <= 120:
            raise ValueError("Age must be between 1 and 120.")

        if not 40 <= trestbps <= 300:
            raise ValueError(
                "Resting blood pressure must be between 40 and 300."
            )

        if not 50 <= chol <= 800:
            raise ValueError(
                "Cholesterol must be between 50 and 800."
            )

        if not 30 <= thalach <= 250:
            raise ValueError(
                "Maximum heart rate must be between 30 and 250."
            )

        if not 0 <= oldpeak <= 20:
            raise ValueError(
                "ST depression must be between 0 and 20."
            )

        # Convert dropdown choices to the dataset's category codes
        sex = 1 if fields["Gender"].get() == "Male" else 0

        cp_map = {
            "Typical Angina": 1,
            "Atypical Angina": 2,
            "Non-anginal Pain": 3,
            "Asymptomatic": 4
        }

        fbs_map = {
            "No": 0,
            "Yes": 1
        }

        restecg_map = {
            "Normal": 0,
            "ST-T Abnormality": 1,
            "Left Ventricular Hypertrophy": 2
        }

        exang_map = {
            "No": 0,
            "Yes": 1
        }

        slope_map = {
            "Upsloping": 1,
            "Flat": 2,
            "Downsloping": 3
        }

        thal_map = {
            "Normal": 3,
            "Fixed Defect": 6,
            "Reversible Defect": 7
        }

        ca = int(fields["Number of Major Vessels (CA)"].get())

        # IMPORTANT: Keep this order identical to the training features.
        patient_data = np.array([[
            age,
            sex,
            cp_map[fields["Chest Pain Type"].get()],
            trestbps,
            chol,
            fbs_map[
                fields["Fasting Blood Sugar > 120 mg/dL"].get()
            ],
            restecg_map[fields["Resting ECG"].get()],
            thalach,
            exang_map[
                fields["Exercise-Induced Angina"].get()
            ],
            oldpeak,
            slope_map[fields["ST Segment Slope"].get()],
            ca,
            thal_map[fields["Thalassemia"].get()]
        ]], dtype=float)

        # Apply the same fitted scaler used during training
        patient_scaled = scaler.transform(patient_data)

        # Predict probability
        probability = float(
            np.asarray(model.predict(patient_scaled, verbose=0)).reshape(-1)[0]
        )

        if not 0 <= probability <= 1:
            raise ValueError(
                "The model returned an invalid probability."
            )

        # Existing decision threshold
        if probability >= 0.5:
            result = "Heart Disease Detected"
            result_color = "#B42318"
        else:
            result = "No Heart Disease Detected"
            result_color = "#067647"

        probability_text = f"Model score: {probability * 100:.2f}%"

        result_label.config(
            text=f"{result}\n{probability_text}",
            fg=result_color
        )

        disclaimer_label.config(
            text=(
                "This is an AI-generated prediction, not a medical "
                "diagnosis. Consult a qualified healthcare professional."
            )
        )

    except ValueError as e:
        messagebox.showerror(
            "Invalid Input",
            str(e) or "Please check the entered values."
        )

    except Exception as e:
        messagebox.showerror(
            "Prediction Error",
            f"Could not make a prediction.\n\nDetails: {e}"
        )


# Predict button
predict_button = tk.Button(
    root,
    text="Predict Heart Disease",
    command=predict,
    font=("Arial", 12, "bold"),
    bg=ACCENT_COLOR,
    fg="white",
    activebackground=HEADER_COLOR,
    activeforeground="white",
    relief="flat",
    cursor="hand2",
    padx=20,
    pady=10
)
predict_button.pack(pady=12)

# Prediction result
result_label = tk.Label(
    root,
    text="Enter the details and click Predict",
    font=("Arial", 13, "bold"),
    bg=BG_COLOR,
    fg=TEXT_COLOR,
    justify="center",
    wraplength=570
)
result_label.pack(pady=8)

# Medical disclaimer
disclaimer_label = tk.Label(
    root,
    text="For educational and research purposes only.",
    font=("Arial", 9),
    bg=BG_COLOR,
    fg="#586574",
    wraplength=570,
    justify="center"
)
disclaimer_label.pack(pady=(0, 12))

root.mainloop()
