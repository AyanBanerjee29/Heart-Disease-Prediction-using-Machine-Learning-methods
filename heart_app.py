import gradio as gr
import pandas as pd
import joblib

# 📦 Load all saved models
models = {
    "Logistic Regression": joblib.load("log_model.pkl"),
    "SVM (Linear)": joblib.load("svm_model.pkl"),
    "Random Forest": joblib.load("rf_model.pkl"),
    "Decision Tree": joblib.load("dt_model.pkl"),
    "Bagging": joblib.load("bagging_model.pkl"),
    "AdaBoost": joblib.load("ada_model.pkl"),
    "Voting Classifier": joblib.load("voting_model.pkl")
}

# 🔍 Prediction function
def predict_heart_disease(model_name, age, sex, chest_pain, bp, chol, sugar, ecg,
                          max_hr, angina, oldpeak, slope, vessels, thal, threshold):
    model = models[model_name]

    # Input dict
    input_data = {
        'age': age,
        'sex': sex,
        'chest_pain_type': chest_pain,
        'resting_blood_pressure': bp,
        'cholestoral': chol,
        'fasting_blood_sugar': sugar,
        'rest_ecg': ecg,
        'Max_heart_rate': max_hr,
        'exercise_induced_angina': angina,
        'oldpeak': oldpeak,
        'slope': slope,
        'vessels_colored_by_flourosopy': vessels,
        'thalassemia': thal
    }

    input_df = pd.DataFrame([input_data])
    proba = model.predict_proba(input_df)[0]
    prob_0 = proba[0]
    prob_1 = proba[1]

    prediction = int(prob_1 > threshold)
    result = "💔 Heart Disease Detected" if prediction == 1 else "❤️ No Heart Disease"

    return f"{result}\n\nProbability of Heart Disease: {prob_1:.2f}\nProbability of No Disease: {prob_0:.2f}"

# 🎛️ Gradio UI
interface = gr.Interface(
    fn=predict_heart_disease,
    inputs=[
        gr.Dropdown(list(models.keys()), label="Select Model"),
        gr.Number(label="Age", value=50),
        gr.Radio(['Male', 'Female'], label="Sex"),
        gr.Dropdown(['typical', 'atypical', 'non-anginal', 'asymptomatic'], label="Chest Pain Type"),
        gr.Number(label="Resting Blood Pressure"),
        gr.Number(label="Cholesterol"),
        gr.Radio(['Greater than 120 mg/ml', 'Lower than 120 mg/ml'], label="Fasting Blood Sugar"),
        gr.Dropdown(['normal', 'ST-T wave abnormality', 'left ventricular hypertrophy'], label="Resting ECG"),
        gr.Number(label="Max Heart Rate"),
        gr.Radio(['Yes', 'No'], label="Exercise Induced Angina"),
        gr.Number(label="Oldpeak"),
        gr.Dropdown(['upsloping', 'flat', 'downsloping'], label="Slope"),
        gr.Dropdown(['Zero', 'One', 'Two', 'Three'], label="Vessels Colored by Flourosopy"),
        gr.Dropdown(['normal', 'fixed defect', 'reversible defect'], label="Thalassemia"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="Decision Threshold (default 0.5)")
    ],
    outputs=gr.Text(label="Prediction Result"),
    title="💓 Heart Disease Prediction App",
    description="Select a model and enter patient information to predict the presence of heart disease.\nYou can also adjust the decision threshold (e.g. 0.6 = be more cautious)."
)

# 🚀 Run the app
interface.launch()
