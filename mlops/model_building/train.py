
import os
import pandas as pd
import joblib
import xgboost as xgb

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError


# Connect to Hugging Face
api = HfApi(token=os.getenv("HF_TOKEN"))


# Load prepared training and testing data
Xtrain = pd.read_csv(
    "hf://datasets/Jags99/Bank-Customer-Churn/Xtrain.csv"
)

Xtest = pd.read_csv(
    "hf://datasets/Jags99/Bank-Customer-Churn/Xtest.csv"
)

ytrain = pd.read_csv(
    "hf://datasets/Jags99/Bank-Customer-Churn/ytrain.csv"
)

ytest = pd.read_csv(
    "hf://datasets/Jags99/Bank-Customer-Churn/ytest.csv"
)


# Features
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

categorical_features = [
    "Geography"
]


# Handle class imbalance
class_weight = (
    ytrain.value_counts()[0] /
    ytrain.value_counts()[1]
)


# Data preprocessing
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)


# Create XGBoost model
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42
)


# Create pipeline
model_pipeline = make_pipeline(
    preprocessor,
    xgb_model
)


# Hyperparameter tuning
param_grid = {
    "xgbclassifier__n_estimators": [50, 100],
    "xgbclassifier__max_depth": [2, 3, 4],
    "xgbclassifier__learning_rate": [0.01, 0.05, 0.1]
}


grid_search = GridSearchCV(
    model_pipeline,
    param_grid,
    cv=5,
    n_jobs=-1
)


# Train model
grid_search.fit(
    Xtrain,
    ytrain.values.ravel()
)


# Select best model
best_model = grid_search.best_estimator_


# Evaluate model
classification_threshold = 0.45

y_pred_train = (
    best_model.predict_proba(Xtrain)[:,1]
    >= classification_threshold
).astype(int)

y_pred_test = (
    best_model.predict_proba(Xtest)[:,1]
    >= classification_threshold
).astype(int)


print("Training Performance")
print(classification_report(ytrain, y_pred_train))

print("Testing Performance")
print(classification_report(ytest, y_pred_test))


# Save model
joblib.dump(
    best_model,
    "best_churn_model.joblib"
)


# Upload model to Hugging Face Model Hub

model_repo_id = "Jags99/churn-model"
model_repo_type = "model"


try:
    api.repo_info(
        repo_id=model_repo_id,
        repo_type=model_repo_type
    )
except RepositoryNotFoundError:
    create_repo(
        repo_id=model_repo_id,
        repo_type=model_repo_type,
        private=False
    )


api.upload_file(
    path_or_fileobj="best_churn_model.joblib",
    path_in_repo="best_churn_model.joblib",
    repo_id=model_repo_id,
    repo_type=model_repo_type
)


print("Model training and upload completed successfully.")
