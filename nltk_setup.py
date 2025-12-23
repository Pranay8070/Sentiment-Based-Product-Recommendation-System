import nltk
import os

# Set the folder path where NLTK data will be downloaded
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NLTK_DIR = os.path.join(BASE_DIR, 'nltk_data')

# Make folder if it doesn't exist
os.makedirs(NLTK_DIR, exist_ok=True)

# Download the required NLTK packages into this folder
nltk.download('stopwords', download_dir=NLTK_DIR)
nltk.download('punkt', download_dir=NLTK_DIR)
nltk.download('averaged_perceptron_tagger', download_dir=NLTK_DIR)
nltk.download('wordnet', download_dir=NLTK_DIR)
nltk.download('omw-1.4', download_dir=NLTK_DIR)
