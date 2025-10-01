# Libraries
import kaggle

kaggle.api.authenticate()

# Check files
print(kaggle.api.dataset_list_files("nih-chest-xrays/data").files)

# Download dataset
kaggle.api.dataset_download_file(
    "nih-chest-xrays/data", "Data_Entry_2017.csv", path="../data/raw", force=True
)
