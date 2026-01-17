
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.absolute()
DB_PATH = str(BASE_DIR / "iso_standards.db")
EMBEDDINGS_PATH = str(BASE_DIR / "embeddings.npy")
EMBEDDINGS_IDS_PATH = str(BASE_DIR / "embeddings_ids.csv")

# App settings
COLLECTION_NAME = "iso_standards"
DOMAIN = "international standardization"

# Model settings (using Groq as per spec, though user key must be provided)
# Ensure you have GROQ_API_KEY in your environment variables.
GROQ_MODEL = "llama-3.3-70b-versatile" 
