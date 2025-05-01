# concept_retriever_openai.py
# Optimized version: Retrieves more candidates first, then filters in Python.

import argparse
import chromadb
import time
import os # Added for isfile check
from openai import OpenAI, RateLimitError, APIError

# --- Configuration and Translation Import ---
try:
    from config_openai import (
        OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL, OPENAI_TRANSLATION_MODEL, # Added translation model here too
        CHROMA_PATH, COLLECTION_NAME
    )
except ImportError:
    print("ERROR: Failed to import configuration from config_openai.py.")
    print("Please ensure the file exists and is in the correct path.")
    exit(1)
except ValueError as e: # Catch missing API key from config
     print(f"ERROR in configuration: {e}")
     exit(1)

# Assuming translator_openai.py is in the same directory or Python path
try:
    if not os.path.isfile("translator_openai.py"):
        raise ImportError("translator_openai.py not found in the current directory.")
    from translator_openai import get_translation_openai
except ImportError as e:
    print(f"ERROR: Could not import translation function: {e}")
    print("Please ensure translator_openai.py exists and is correctly defined.")
    exit(1)
# --- End Imports ---


# --- Initialize OpenAI Client ---
# Used for both embeddings and potentially translation
try:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    print("OpenAI client initialized successfully.")
except Exception as e:
    print(f"ERROR: Failed to initialize OpenAI client: {e}")
    exit(1)
# --- End OpenAI Init ---


# --- ChromaDB Client will be initialized in find_concepts_openai function ---
chroma_client = None
collection = None

# --- Helper Functions ---
def get_query_embedding(text: str, model: str) -> list[float] | None:
    """Generates embedding for the query text using OpenAI."""
    print(f"    Generating embedding for query using model: {model}...")
    try:
        response = openai_client.embeddings.create(input=[text], model=model)
        embedding = response.data[0].embedding
        print("    Embedding generated successfully.")
        return embedding
    except (RateLimitError, APIError) as e:
        print(f"ERROR: OpenAI API error during embedding generation ({type(e).__name__}). Cannot proceed.")
        return None
    except Exception as e:
        print(f"ERROR: Unexpected error during embedding generation: {e}")
        return None

# --- Main Retrieval Logic ---
def find_concepts_openai(query: str, domain: str, vocabulary: str | None = None, k: int = 5, db_domain: str = None):
    """
    Finds candidate OMOP concepts using OpenAI embeddings.
    Optimized: Retrieves more candidates first, then filters in Python.
    """
    overall_start_time = time.time()
    
    # Initialize ChromaDB Client with the appropriate path based on domain
    global chroma_client, collection
    # If db_domain is specified, use it instead of the filter domain
    path_domain = db_domain if db_domain is not None else domain
    actual_chroma_path = os.path.join(CHROMA_PATH, path_domain)
    
    try:
        print(f"\nConnecting to ChromaDB at: {actual_chroma_path}")
        if not os.path.exists(actual_chroma_path):
            print(f"ERROR: ChromaDB path does not exist: {actual_chroma_path}")
            print(f"Please ensure the index was built correctly first for domain: {domain}")
            exit(1)

        chroma_client = chromadb.PersistentClient(path=actual_chroma_path)
        # Verify collection exists before proceeding
        print(f"Getting collection: {COLLECTION_NAME}")
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
        collection_count = collection.count() # Get count early
        print(f"Successfully connected to ChromaDB collection '{COLLECTION_NAME}' at '{actual_chroma_path}'.")
        print(f"Collection contains approx. {collection_count} concepts.")
        if collection_count == 0:
            print("WARNING: The ChromaDB collection is empty. Retrieval will yield no results.")
    except Exception as e:
        print(f"ERROR: Could not connect to ChromaDB collection '{COLLECTION_NAME}' at '{actual_chroma_path}'.")
        print(f"Details: {e}")
        print(f"Please ensure the index was built correctly for domain: {domain}")
        exit(1)

    print("\n--- Starting Concept Search ---")

    # 1. Translate query
    start_time = time.time()
    print(f"  Translating query: '{query}'...")
    # ***** CORRECTED CALL *****
    # Call the function as it's likely defined in translator_openai.py,
    # assuming it handles client/model internally or via config.
    translated_query = get_translation_openai(
        text=query,
        target_language='en'
        )
    # ***** END CORRECTION *****

    if not translated_query or translated_query == query:
        print("  Proceeding with original/untranslated query.")
        translated_query = query # Ensure it's not None
    else:
        print(f"  Translated query: '{translated_query}'")
    print(f"  Translation took {time.time() - start_time:.2f} seconds.")

    # 2. Generate query embedding
    start_time = time.time()
    query_embedding = get_query_embedding(translated_query, OPENAI_EMBEDDING_MODEL)
    if query_embedding is None:
        return [] # Cannot search without embedding
    print(f"  Embedding generation took {time.time() - start_time:.2f} seconds.")

    # 3. Query ChromaDB *without* metadata filter, requesting more results
    start_time = time.time()
    initial_k_multiplier = 2000
    initial_k = k * initial_k_multiplier
    print(f"  Querying ChromaDB (initial k={initial_k}, no metadata filter)...")
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=initial_k,
            where=None, # No filter applied at DB level
            include=['metadatas', 'distances'] # Get metadata and distance score
        )
        print(f"  Initial ChromaDB query took {time.time() - start_time:.2f} seconds.")

    except Exception as e:
        print(f"ERROR: Error during initial ChromaDB query: {e}")
        traceback.print_exc()
        return []

    # 4. Filter results in Python
    start_time = time.time()
    num_retrieved = len(results['ids'][0]) if results and results.get('ids') else 0
    print(f"  Filtering {num_retrieved} candidates in Python...")
    filtered_candidates = []
    target_domain_lower = domain.lower() if domain else None
    target_vocabulary_lower = vocabulary.lower() if vocabulary else None

    if num_retrieved > 0:
        ids = results['ids'][0]
        distances = results['distances'][0]
        metadatas = results['metadatas'][0]

        for concept_id_str, dist, meta in zip(ids, distances, metadatas):
            # Apply filters (domain, vocabulary) - case-insensitive
            passes_domain = (not target_domain_lower or
                             (isinstance(meta.get('domain_id'), str) and meta['domain_id'].lower() == target_domain_lower))
            passes_vocabulary = (not target_vocabulary_lower or
                                 (isinstance(meta.get('vocabulary_id'), str) and meta['vocabulary_id'].lower() == target_vocabulary_lower))

            if passes_domain and passes_vocabulary:
                filtered_candidates.append({
                    'concept_id': meta.get('concept_id', 'N/A'),
                    'concept_name': meta.get('concept_name', 'N/A'),
                    'domain_id': meta.get('domain_id', 'N/A'),
                    'vocabulary_id': meta.get('vocabulary_id', 'N/A'),
                    'score': float(dist)
                })

            if len(filtered_candidates) >= k: break # Stop early
    else:
        print("  No initial results found from vector search.")

    print(f"  Python filtering took {time.time() - start_time:.2f} seconds.")

    # 5. Return the top K filtered results
    end_time = time.time()
    print(f"--- Total Search Process Took {end_time - overall_start_time:.2f} seconds ---")
    return filtered_candidates[:k]

# --- Display Function ---
def display_results(candidates):
    """Formats and prints the candidate concepts."""
    if not candidates:
        print("\nNo matching standard OMOP concepts found for the specified criteria.")
        return

    print("\nPotential Standard OMOP Concept Matches:")
    print("-" * 70)
    # Results are sorted by distance (lower is better) from ChromaDB query
    for i, concept in enumerate(candidates):
        print(f"{i+1}. Concept ID:   {concept['concept_id']}")
        print(f"   Concept Name: {concept['concept_name']}")
        print(f"   Domain ID:    {concept['domain_id']}")
        print(f"   Vocabulary:   {concept['vocabulary_id']}")
        # Lower score is better (distance)
        print(f"   Score (Distance): {concept['score']:.4f} (Lower score indicates better match)")
        print("-" * 70)

# --- Main Execution Block ---
if __name__ == "__main__":
    # Import necessary libraries for main execution, like argparse, traceback
    import argparse
    import traceback # Ensure traceback is imported here too

    parser = argparse.ArgumentParser(description="Retrieve OMOP Concepts using RAG with OpenAI Embeddings.")
    parser.add_argument("query", type=str, help="The input word, phrase, or description.")
    parser.add_argument("domain", type=str, help="The target OMOP domain to filter by (e.g., Condition, Drug). Also used to determine the ChromaDB path by default.")
    parser.add_argument("-v", "--vocabulary", type=str, default=None, help="(Optional) The target OMOP vocabulary to filter by (e.g., SNOMED, RxNorm). Case-insensitive.")
    parser.add_argument("-k", "--top_k", type=int, default=5, help="Number of results to return.")
    parser.add_argument("--use-all-db", action="store_true", help="Use the default 'All' ChromaDB path instead of domain-specific path")

    args = parser.parse_args()

    print("\n--- OMOP Concept Retriever (OpenAI + ChromaDB) ---")
    print(f"Query: '{args.query}'")
    print(f"Target Domain: {args.domain}")
    if args.vocabulary:
        print(f"Target Vocabulary: {args.vocabulary}")
    print(f"Number of Results (k): {args.top_k}")
    
    # Determine which ChromaDB path to use
    db_domain = "All" if args.use_all_db else args.domain
    if args.use_all_db:
        print("Using 'All' ChromaDB path (combined database) with domain filter")
    else:
        print(f"Using domain-specific ChromaDB path: {db_domain}")

    try:
        candidates = find_concepts_openai(args.query, args.domain, args.vocabulary, args.top_k, db_domain=db_domain)
        display_results(candidates)
    except Exception as e:
         print("\n--- An Unexpected Error Occurred During Retrieval ---")
         print(f"Error Type: {type(e).__name__}")
         print(f"Error Details: {e}")
         print("Traceback:")
         traceback.print_exc()
         print("----------------------------------------------------")

    print("\n--- Retrieval Script Finished ---")