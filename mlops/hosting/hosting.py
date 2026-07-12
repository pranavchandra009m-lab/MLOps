
import os
from huggingface_hub import HfApi

api = HfApi(token=os.getenv("HF_TOKEN"))

api.upload_folder(
    folder_path="mlops/deployment",
    repo_id="Jags99/Bank-Customer-Churn",
    repo_type="space",
    path_in_repo=""
)

print("Deployment files uploaded successfully.")
