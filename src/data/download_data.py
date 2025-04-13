import os
import requests
import zipfile
from tqdm import tqdm

def download_file(url: str, filename: str):
    """
    Download a file from a URL with progress bar.
    
    Args:
        url (str): URL to download from
        filename (str): Local filename to save to
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    
    with open(filename, 'wb') as f, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            size = f.write(data)
            bar.update(size)

def download_movielens_data(data_dir: str = "data"):
    """
    Download and extract the MovieLens dataset.
    
    Args:
        data_dir (str): Directory to save the data
    """
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Download the dataset
    url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    zip_path = os.path.join(data_dir, "ml-latest-small.zip")
    
    print("Downloading MovieLens dataset...")
    download_file(url, zip_path)
    
    # Extract the dataset
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    
    # Remove the zip file
    os.remove(zip_path)
    print("Dataset downloaded and extracted successfully!")

if __name__ == "__main__":
    download_movielens_data() 