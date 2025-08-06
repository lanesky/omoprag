# OMOP Concept Retriever with OpenAI and ChromaDB

## Project File Overview

### Configuration Files

#### `config_openai.py`
- Contains all project configuration parameters
- Sets up OpenAI API key and model selection
- Configures data file paths (CONCEPT.csv)
- Configures ChromaDB storage path and collection name
- Handles environment variable loading and basic validation

### Index Building

#### `vocabulary_indexer_openai_streaming.py`
- Improved index builder with support for large datasets
- Processes CONCEPT.csv file in chunks
- Supports filtering concepts by specific domains (e.g., Drug) via command line arguments
- Dynamically sets ChromaDB path based on domain
- Includes robust error handling and retry logic
- Provides detailed progress and performance logging

### Concept Retrieval

#### `concept_retriever_openai.py`
- Main concept retrieval tool
- Accepts user queries (supports Chinese, Japanese, or English)
- Uses OpenAI API to translate queries to English if needed
- Generates query embeddings
- Searches for similar concepts in ChromaDB
- Supports filtering results by domain and vocabulary
- Supports searching in all data (All) or specific domain databases
- Formats and displays retrieval results

#### `concept_retriever_openai_quick.py`
- Optimized concept retrieval tool
- Retrieves more candidate concepts first, then filters in Python
- Provides faster retrieval and better result quality
- Supports searching in all data (All) or specific domain databases
- Enhanced error handling and user feedback

### Utility Functions

#### `translator_openai.py`
- Provides text translation functionality
- Uses OpenAI API to translate medical terms and related text to target languages
- Specifically designed for accurate translation of medical terminology
- Includes error handling and fallback mechanisms

## Usage Flow

1. Configure parameters in `config_openai.py`
2. Build concept index using `vocabulary_indexer_openai_streaming.py` (optional: for specific domains)
3. Retrieve concepts using `concept_retriever_openai.py` or `concept_retriever_openai_quick.py`

## Examples

```bash
# Build index for all concepts (stored in All directory)
python vocabulary_indexer_openai_streaming.py

# Build index only for Drug domain concepts (stored in Drug directory)
python vocabulary_indexer_openai_streaming.py Drug

# Search for concepts in a specific domain database
python concept_retriever_openai.py "hypertension" Condition

# Search for concepts in all data (All database) with domain filtering
# (Advantage: matches semantics and domain across all data with high precision.
# Disadvantage: Slower, with queries potentially taking over 3 minutes)
python concept_retriever_openai.py "hypertension" Condition --use-all-db

# Use optimized retrieval tool (faster as it first matches concept_name semantically, then filters by domain.
# Disadvantage: When using `--use-all-db`, if candidate concepts don't include matching domains, results may be less accurate)
python concept_retriever_openai_quick.py "hypertension" Condition
```

## Data Preparation and Index Building Process

### 1. Obtain OMOP Concept Data

1. Download OMOP Vocabulary data from [ATHENA](https://athena.ohdsi.org/)
   - Register and log in to the ATHENA website
   - Download the latest version of the Vocabulary package
   - Extract the package to get the CONCEPT.csv file

2. (Optional) Import CONCEPT.csv into a PostgreSQL database for processing
   ```sql
   CREATE TABLE concept (
     concept_id INTEGER PRIMARY KEY,
     concept_name VARCHAR(255),
     domain_id VARCHAR(20),
     vocabulary_id VARCHAR(20),
     concept_class_id VARCHAR(20),
     standard_concept VARCHAR(1),
     concept_code VARCHAR(50),
     valid_start_date DATE,
     valid_end_date DATE,
     invalid_reason VARCHAR(1)
   );
   
   COPY concept FROM '/path/to/CONCEPT.csv' DELIMITER E'\t' CSV HEADER;
   ```

### 2. Obtain CPT4 Data (Optional)

1. Apply for a CPT4 License Key
   - Visit [UMLS website](https://uts.nlm.nih.gov/uts/login)
   - Follow the instructions to apply for a CPT4 license

2. Download CPT4 data using the License Key

3. Merge CPT4 data into the CONCEPT.csv file

### 3. Prepare Index Data

1. Filter standard concepts from the complete CONCEPT.csv
   - Use SQL or pandas for filtering
   - Save to `csv_for_indexing/CONCEPT.csv`

2. Configure the `config_openai.py` file
   - Set up OpenAI API key
   - Configure data file paths and ChromaDB storage paths

### 4. Build the Index

1. Build index for all concepts using `vocabulary_indexer_openai_streaming.py`
   ```bash
   python vocabulary_indexer_openai_streaming.py
   ```

2. Or build index for specific domains
   ```bash
   python vocabulary_indexer_openai_streaming.py Drug
   python vocabulary_indexer_openai_streaming.py Condition
   python vocabulary_indexer_openai_streaming.py Procedure
   # ... etc.
   ```

The index building process may take some time depending on the size of your concept data and your OpenAI API limits. Once the index is built, you can start using the concept retrieval tools to perform queries.
