# vocabulary_indexer_openai.py
import pandas as pd
import os
import time
import numpy as np
from openai import OpenAI, RateLimitError, APIError
import chromadb

# Import configuration
from config_openai import (
    OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL, OPENAI_EMBEDDING_BATCH_SIZE,
    CONCEPT_FILE, CHROMA_PATH, COLLECTION_NAME
)

# Initialize OpenAI Client
client = OpenAI(api_key=OPENAI_API_KEY)

def load_omop_concepts(concept_file_path):
    """Loads standard concepts from the CONCEPT.csv file."""
    print(f"Loading concepts from {concept_file_path}...")
    try:
        dtype_spec = {
            'concept_id': int,
            'concept_name': str,
            'domain_id': str,
            'vocabulary_id': str,
            'concept_class_id': str,
            'standard_concept': str,
            'concept_code': str,
            'valid_start_date': str,
            'valid_end_date': str,
            'invalid_reason': str
        }
        usecols = ['concept_id', 'concept_name', 'domain_id', 'vocabulary_id', 'standard_concept']

        df = pd.read_csv(
            concept_file_path,
            sep='\t', # Adjust if your separator is different
            dtype=dtype_spec,
            usecols=usecols,
            on_bad_lines='warn',
            quoting=3 # csv.QUOTE_NONE
        )

        # Filter for Standard Concepts with non-empty names
        df_standard = df[(df['standard_concept'] == 'S') & (df['concept_name'].notna()) & (df['concept_name'].str.strip() != '')].copy()

        print(f"Loaded {len(df_standard)} standard concepts with valid names.")
        return df_standard

    except FileNotFoundError:
        print(f"Error: Concept file not found at {concept_file_path}")
        return None
    except Exception as e:
        print(f"Error loading or processing concept file: {e}")
        return None

def generate_openai_embeddings_batch(texts: list[str], model: str, batch_size: int) -> list | None:
    """Generates OpenAI embeddings for a list of texts in batches with retry logic."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:min(i + batch_size, len(texts))]
        print(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}...")

        # Simple retry mechanism
        retries = 3
        delay = 5 # seconds
        while retries > 0:
            try:
                response = client.embeddings.create(
                    input=batch,
                    model=model
                )
                embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(embeddings)
                print(f"  Successfully embedded {len(batch)} texts.")
                time.sleep(1) # Small delay to help with rate limits
                break # Success, exit retry loop
            except RateLimitError:
                print(f"  Rate limit hit. Retrying in {delay} seconds... ({retries-1} retries left)")
                time.sleep(delay)
                retries -= 1
                delay *= 2 # Exponential backoff
            except APIError as e:
                 print(f"  OpenAI API Error: {e}. Retrying in {delay} seconds... ({retries-1} retries left)")
                 time.sleep(delay)
                 retries -=1
                 delay *=2
            except Exception as e:
                print(f"  An unexpected error occurred during embedding: {e}")
                # Decide if you want to retry on generic errors or fail
                retries = 0 # Stop retrying on unexpected errors for this batch
                # Optionally: return None or raise error to stop the whole process

        if retries == 0:
            print(f"Error: Failed to embed batch starting at index {i} after multiple retries.")
            return None # Indicate failure for the whole process

    return all_embeddings


def build_index_chroma_openai(concepts_df):
    """Builds ChromaDB index using OpenAI embeddings."""
    if concepts_df is None or concepts_df.empty:
        print("Cannot build index: No concepts data.")
        return False

    # 1. Get texts to embed
    concept_names = concepts_df['concept_name'].astype(str).tolist()
    print(f"Total concept names to embed: {len(concept_names)}")

    # 2. Generate Embeddings using OpenAI (batched)
    print(f"Generating embeddings using OpenAI model: {OPENAI_EMBEDDING_MODEL}...")
    start_time = time.time()
    embeddings = generate_openai_embeddings_batch(
        texts=concept_names,
        model=OPENAI_EMBEDDING_MODEL,
        batch_size=OPENAI_EMBEDDING_BATCH_SIZE
    )
    end_time = time.time()

    if embeddings is None:
        print("Embedding generation failed. Index building aborted.")
        return False
    if len(embeddings) != len(concepts_df):
        print(f"Error: Number of embeddings ({len(embeddings)}) does not match number of concepts ({len(concepts_df)}). Aborting.")
        return False

    print(f"Embedding generation completed in {end_time - start_time:.2f} seconds.")

    # 3. Setup ChromaDB
    print(f"Setting up ChromaDB client at {CHROMA_PATH}...")
    # Using PersistentClient to save data to disk
    client_chroma = chromadb.PersistentClient(path=CHROMA_PATH)

    print(f"Creating or getting collection: {COLLECTION_NAME}")
    # When adding embeddings directly, we don't specify an embedding function for the collection
    # If the collection exists and used a different embedding method, this might cause issues.
    # It's often best to delete an old collection if changing embedding methods.
    try:
        collection = client_chroma.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"} # Use cosine similarity
        )
    except Exception as e:
        print(f"Error creating/getting ChromaDB collection: {e}")
        print("Consider deleting the existing index directory if changing configurations.")
        return False

    # 4. Add data to ChromaDB in batches
    print("Adding data to ChromaDB collection...")
    add_batch_size = 5000 # Batch size for adding to ChromaDB (can be different from embedding batch)
    num_batches = int(np.ceil(len(concepts_df) / add_batch_size))
    total_added = 0

    for i in range(num_batches):
        start_idx = i * add_batch_size
        end_idx = min((i + 1) * add_batch_size, len(concepts_df))
        batch_df = concepts_df.iloc[start_idx:end_idx]
        batch_embeddings = embeddings[start_idx:end_idx]

        ids = batch_df['concept_id'].astype(str).tolist() # Chroma IDs must be strings
        # We are providing embeddings directly, so 'documents' is optional but good practice
        documents = batch_df['concept_name'].astype(str).tolist()
        metadatas = batch_df[['concept_id', 'domain_id', 'vocabulary_id', 'concept_name']].to_dict('records')
        # Ensure concept_id in metadata is int if needed later
        for meta in metadatas:
             meta['concept_id'] = int(meta['concept_id'])

        print(f"Adding batch {i+1}/{num_batches} ({len(ids)} items) to ChromaDB...")
        try:
            collection.add(
                ids=ids,
                embeddings=batch_embeddings,
                documents=documents, # Optional but recommended
                metadatas=metadatas
            )
            total_added += len(ids)
        except Exception as e:
            print(f"Error adding batch {i+1} to ChromaDB: {e}")
            # Consider adding retry logic or logging failed batches/IDs

    print("-" * 30)
    print(f"Index building complete.")
    print(f"Total concepts processed: {len(concepts_df)}")
    print(f"Total items added to ChromaDB collection '{COLLECTION_NAME}': {total_added} (approx count: {collection.count()})")
    print(f"Index data stored at: {CHROMA_PATH}")
    print("-" * 30)
    return True

def run_openai_indexing():
    """Loads data and builds the index using OpenAI embeddings."""
    if not OPENAI_API_KEY:
        print("Error: OpenAI API Key not configured. Set the OPENAI_API_KEY environment variable.")
        return

    concepts = load_omop_concepts(CONCEPT_FILE)
    if concepts is not None:
        build_index_chroma_openai(concepts)

if __name__ == "__main__":
    run_openai_indexing()