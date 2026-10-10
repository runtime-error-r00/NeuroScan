import shutil
from pathlib import Path
import kagglehub

path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
print("downloaded in:", path)

dest = Path(__file__).resolve().parent.parent / "data" / "raw"
shutil.copytree(path, dest, dirs_exist_ok=True)
print("copied to:", dest)