import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

repo_id = "Jags99/Bank-Customer-Churn"
repo_type = "dataset"

api = HfApi(token=os.getenv("HF_TOKEN"))

# Create dataset repository if it does not exist
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Repository {repo_id} already exists.")
except RepositoryNotFoundError:
    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=False
    )
    print(f"Repository {repo_id} created.")

# Upload dataset files
api.upload_folder(
    folder_path="mlops/data",
    repo_id=repo_id,
    repo_type=repo_type
)

print("Dataset uploaded successfully.")
