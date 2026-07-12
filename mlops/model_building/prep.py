
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

# Hugging Face connection
api = HfApi(token=os.getenv("HF_TOKEN"))

# Dataset location on Hugging Face
DATASET_PATH = "hf://datasets/Jags99/Bank-Customer-Churn/bank_customer_churn.csv"

# Load dataset
bank_dataset = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully.")

# Target variable (what we want to predict)
target = "Exited"

# Numerical columns
numeric_features = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary"
]

# Categorical columns
categorical_features = [
    "Geography"
]

# Input features
X = bank_dataset[numeric_features + categorical_features]

# Output variable
y = bank_dataset[target]

# Split data into training and testing sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Save split data locally
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

# Upload split data back to Hugging Face dataset repository

files = [
    "Xtrain.csv",
    "Xtest.csv",
    "ytrain.csv",
    "ytest.csv"
]

for file in files:
    api.upload_file(
        path_or_fileobj=file,
        path_in_repo=file,
        repo_id="Jags99/Bank-Customer-Churn",
        repo_type="dataset"
    )

print("Data preparation completed successfully.")
