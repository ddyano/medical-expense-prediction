import pandas as pd
import joblib

print("Prediction environment is ready!")


model = joblib.load("models/gradient_boosting_model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")

print("Model and preprocessor loaded successfully!")

# validating features
gender = input("Enter gender (male/female): ").strip().lower()

if gender not in ["male", "female"]:
    print("Invalid gender. Please enter male or female.")
    exit()
region = input(
    "Enter region (northeast/northwest/southeast/southwest): "
).strip().lower()

if region not in ["northeast", "northwest", "southeast", "southwest"]:
    print("Invalid region. Please enter northeast, northwest, southeast, or southwest.")
    exit()

smoker = input(
    "Enter smoker status (smoker/non-smoker): "
).strip().lower()

if smoker not in ["smoker", "non-smoker"]:
    print("Invalid smoker status. Please enter smoker or non-smoker.")
    exit()
age = int(input("Enter age: "))

if age < 18 or age > 100:
    print("Invalid age. Please enter an age between 18 and 100.")
    exit()

bmi = float(input("Enter BMI: "))

if bmi < 10 or bmi > 70:
    print("Invalid BMI. Please enter a realistic BMI value.")
    exit()

children = int(input("Enter number of children: "))

if children < 0 or children > 10:
    print("Invalid number of children. Please enter a value between 0 and 10.")
    exit()


# creating a sample patient
patient = pd.DataFrame({
    "age": [age],
    "gender": [gender],
    "bmi": [bmi],
    "children": [children],
    "region": [region],
    "smoker": [smoker]
})

print(patient)

# preprocessing the patient data
patient_processed = preprocessor.transform(patient)

print("Patient data preprocessed successfully!")
print("Processed shape:", patient_processed.shape)

# making the prediction
prediction = model.predict(patient_processed)

print(f"Predicted medical expense: ₹{prediction[0]:.2f}")