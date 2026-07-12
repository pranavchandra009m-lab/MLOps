
import gradio as gr
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# Download the trained model
model_path = hf_hub_download(
    repo_id="Jags99/churn-model",
    filename="best_churn_model.joblib"
)

# Load the model
model = joblib.load(model_path)


def predict(
    CreditScore,
    Geography,
    Age,
    Tenure,
    Balance,
    NumOfProducts,
    HasCrCard,
    IsActiveMember,
    EstimatedSalary
):

    input_data = pd.DataFrame([{
        "CreditScore": CreditScore,
        "Geography": Geography,
        "Age": Age,
        "Tenure": Tenure,
        "Balance": Balance,
        "NumOfProducts": NumOfProducts,
        "HasCrCard": HasCrCard,
        "IsActiveMember": IsActiveMember,
        "EstimatedSalary": EstimatedSalary
    }])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        return "Customer is likely to churn."
    else:
        return "Customer is not likely to churn."


app = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Credit Score"),
        gr.Dropdown(["France", "Germany", "Spain"], label="Geography"),
        gr.Number(label="Age"),
        gr.Number(label="Tenure"),
        gr.Number(label="Balance"),
        gr.Number(label="Number of Products"),
        gr.Number(label="Has Credit Card (0 or 1)"),
        gr.Number(label="Is Active Member (0 or 1)"),
        gr.Number(label="Estimated Salary")
    ],
    outputs="text",
    title="Bank Customer Churn Prediction"
)

app.launch()
