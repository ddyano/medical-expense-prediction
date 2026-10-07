import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go

st.set_page_config(
    page_title="Medical Expense Prediction",
    page_icon="💰"
)

page = st.sidebar.radio(
    "Navigation",
    ["Prediction", "Model Evaluation"]
)

BASE_DIR = Path(__file__).resolve().parent.parent

linear_model = joblib.load(BASE_DIR / "models" / "linear_regression_model.pkl")
tree_model = joblib.load(BASE_DIR / "models" / "decision_tree_model.pkl")
rf_model = joblib.load(BASE_DIR / "models" / "random_forest_model.pkl")
gb_model = joblib.load(BASE_DIR / "models" / "gradient_boosting_baseline_model.pkl")
best_gb_model = joblib.load(BASE_DIR / "models" / "gradient_boosting_model.pkl")

preprocessor = joblib.load(BASE_DIR / "models" / "preprocessor.pkl")


#evaluation-data code


evaluation_df = pd.read_csv(
    BASE_DIR / "data" / "processed" / "medical_insurance_cleaned.csv"
)

X_eval = evaluation_df.drop("expenses", axis=1)
y_eval = evaluation_df["expenses"]

_, X_test_eval, _, y_test_eval = train_test_split(
    X_eval,
    y_eval,
    test_size=0.2,
    random_state=42
)

X_test_eval_processed = preprocessor.transform(X_test_eval)
model = best_gb_model


st.set_page_config(
    page_title="Medical Expense Prediction",
    page_icon="💰"
)

### First Page========================--=-=-=-==-=-=-=
if page == "Prediction":
    st.title("Medical Expense Prediction")
    st.write("Enter patient details to predict estimated medical expenses.")
    print("Streamlit app started!")
    
    
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30,
        step=1
    )
    
    gender = st.selectbox(
        "Gender",
        ["male", "female"]
    )
    
    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=70.0,
        value=25.5,
        step=0.1
    )
    
    children = st.number_input(
        "Number of Children",
        min_value=0,
        max_value=10,
        value=1,
        step=1
    )
    
    region = st.selectbox(
        "Region",
        ["northeast", "northwest", "southeast", "southwest"]
    )
    
    smoker = st.selectbox(
        "Smoker Status",
        ["non-smoker", "smoker"]
    )
    
    if st.button("Predict Medical Expense"):
        patient = pd.DataFrame({
            "age": [age],
            "gender": [gender],
            "bmi": [bmi],
            "children": [children],
            "region": [region],
            "smoker": [smoker]
        })
    
        st.write(patient)
    
        patient_processed = preprocessor.transform(patient)
        prediction = model.predict(patient_processed)
        st.success(f"Estimated Medical Expense: ₹{prediction[0]:,.2f}")



## Second page =========================!@#%^&%*&^&(*)(*)((_(_)+))
if page == "Model Evaluation":
    st.title("Model Evaluation")
    st.write("Comparison of the regression models used for medical expense prediction.")
    results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting",
        "Tuned Gradient Boosting"
    ],
    "MAE": [
        4412.093860,
        3667.589887,
        2844.805035,
        2795.418782,
        2785.111756
    ],
    "RMSE": [
        6398.138011,
        7270.494413,
        5276.285416,
        5158.732709,
        5134.985414
    ],
    "R²": [
        0.688008,
        0.597131,
        0.787826,
        0.797175,
        0.799038
    ],
    "Within ±10% (%)": [
        15.41,
        65.04,
        46.99,
        40.23,
        32.71
        ]
    })

    st.subheader("Model Performance Comparison")
    st.dataframe(results, use_container_width=True)

    ### the model comparison bar charts
    st.subheader("Model Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.write("MAE")
        st.bar_chart(results.set_index("Model")["MAE"])

    with col2:
        st.write("RMSE")
        st.bar_chart(results.set_index("Model")["RMSE"])

    st.write("R² Score")
    st.bar_chart(results.set_index("Model")["R²"])

        
        
    st.subheader("Actual vs Predicted")

    predictions = {
    "Linear Regression": linear_model.predict(X_test_eval_processed),
    "Decision Tree": tree_model.predict(X_test_eval_processed),
    "Random Forest": rf_model.predict(X_test_eval_processed),
    "Gradient Boosting": gb_model.predict(X_test_eval_processed),
    "Tuned Gradient Boosting": best_gb_model.predict(X_test_eval_processed)
    }

    for model_name, pred in predictions.items():
        st.write(model_name)

        chart_data = pd.DataFrame({
            "Actual": y_test_eval.values,
            "Predicted": pred
        })

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=chart_data["Actual"],
            y=chart_data["Predicted"],
            mode="markers",
            name="Predictions"
        ))

        min_value = min(
            chart_data["Actual"].min(),
            chart_data["Predicted"].min()
        )

        max_value = max(
            chart_data["Actual"].max(),
            chart_data["Predicted"].max()
        )

        fig.add_trace(go.Scatter(
            x=[min_value, max_value],
            y=[min_value, max_value],
            mode="lines",
            name="Perfect Prediction",
            line=dict(dash="dash")
        ))

        fig.update_layout(
            xaxis_title="Actual Medical Expenses",
            yaxis_title="Predicted Medical Expenses",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Residual Analysis")

    residuals = {
        "Linear Regression": y_test_eval.values - predictions["Linear Regression"],
        "Decision Tree": y_test_eval.values - predictions["Decision Tree"],
        "Random Forest": y_test_eval.values - predictions["Random Forest"],
        "Gradient Boosting": y_test_eval.values - predictions["Gradient Boosting"],
        "Tuned Gradient Boosting": y_test_eval.values - predictions["Tuned Gradient Boosting"]
    }

    for model_name, residual in residuals.items():
        st.write(model_name)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=predictions[model_name],
            y=residual,
            mode="markers",
            name="Residuals"
        ))

        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="green"
        )

        fig.update_layout(
            xaxis_title="Predicted Medical Expenses",
            yaxis_title="Residuals (Actual - Predicted)",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)


        st.subheader("Final Model Results")

    final_results = pd.DataFrame([
        {
            "Model": "Linear Regression",
            "MAE": 4412.093860,
            "RMSE": 6398.138011,
            "R2": 0.688008
        },
        {
            "Model": "Decision Tree",
            "MAE": 3667.589887,
            "RMSE": 7270.494413,
            "R2": 0.597131
        },
        {
            "Model": "Random Forest",
            "MAE": 2844.805035,
            "RMSE": 5276.285416,
            "R2": 0.787826
        },
        {
            "Model": "Gradient Boosting",
            "MAE": 2795.418782,
            "RMSE": 5158.732709,
            "R2": 0.797175
        },
        {
            "Model": "Tuned Gradient Boosting",
            "MAE": 2785.111756,
            "RMSE": 5134.985414,
            "R2": 0.799038
        }
    ])

    st.dataframe(final_results, use_container_width=True)

    st.subheader("Best Model")

    st.write("Tuned Gradient Boosting was selected as the final model because it achieved:")

    st.write("• Lowest MAE: ₹2,785.11")
    st.write("• Lowest RMSE: ₹5,134.99")
    st.write("• Highest R² Score: 0.7990")

    st.success("Final Selected Model: Tuned Gradient Boosting")

    