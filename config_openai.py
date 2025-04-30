# config_openai.py
import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables from a .env file if you have one

# --- OpenAI Configuration ---
# IMPORTANT: Store your API key securely, e.g., in an environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

# Choose your desired OpenAI embedding model
# Recommended: "text-embedding-3-small" (cost-effective, good performance)
# Other options: "text-embedding-3-large", "text-embedding-ada-002" (older)
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
# Check OpenAI documentation for maximum batch size for your chosen model
OPENAI_EMBEDDING_BATCH_SIZE = 1000 # Adjust based on model and your rate limits

# OpenAI model for translation tasks
# Options: "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-4.1-nano"
OPENAI_TRANSLATION_MODEL = "gpt-4.1-nano" # Default translation model

# --- Data Configuration ---
VOCABULARY_DIR = 'csv_for_indexing' # Directory containing CONCEPT.csv
CONCEPT_FILE = os.path.join(VOCABULARY_DIR, 'CONCEPT_STANDARD_100.csv')

# --- ChromaDB Configuration ---
CHROMA_PATH = os.path.join('index_openai', 'chroma_db') # Directory to store ChromaDB data
COLLECTION_NAME = "omop_concepts_openai"

# Ensure index directory exists
os.makedirs(os.path.dirname(CHROMA_PATH), exist_ok=True)