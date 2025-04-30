# concept_retriever_openai.py
import argparse
import chromadb
import time
from openai import OpenAI, RateLimitError, APIError

# Import configuration and translation function
from config_openai import (
    OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL,
    CHROMA_PATH, COLLECTION_NAME
)
# Assuming translator_openai.py is in the same directory
try:
    from translator_openai import get_translation_openai
except ImportError:
    print("Error: Could not import get_translation_openai from translator_openai.py.")
    print("Please ensure translator_openai.py exists and is in the same directory.")
    exit(1)


# Initialize OpenAI Client for embeddings
if not OPENAI_API_KEY:
    print("Error: OpenAI API Key not configured.")
    exit(1)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize ChromaDB Client
try:
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
    print(f"Successfully connected to ChromaDB collection '{COLLECTION_NAME}' at '{CHROMA_PATH}'.")
    print(f"Collection contains approx. {collection.count()} concepts.")
except Exception as e:
    print(f"Error: Could not connect to ChromaDB collection '{COLLECTION_NAME}' at '{CHROMA_PATH}'.")
    print(f"Details: {e}")
    print("Please ensure the index was built correctly using vocabulary_indexer_openai.py.")
    exit(1)

def get_query_embedding(text: str, model: str) -> list[float] | None:
    """Generates embedding for the query text using OpenAI."""
    print(f"Generating embedding for query using model: {model}...")
    try:
        response = openai_client.embeddings.create(input=[text], model=model)
        embedding = response.data[0].embedding
        print("Embedding generated successfully.")
        return embedding
    except (RateLimitError, APIError) as e:
        print(f"Error: OpenAI API error during embedding generation ({type(e).__name__}). Cannot proceed.")
        return None
    except Exception as e:
        print(f"Error: Unexpected error during embedding generation: {e}")
        return None

def find_concepts_openai(query: str, domain: str, vocabulary: str | None = None, k: int = 5):
    """
    Finds candidate OMOP concepts using OpenAI embeddings and ChromaDB filtering.
    """
    start_time = time.time()

    # 1. Translate query
    translated_query = get_translation_openai(query, target_language='en')
    if not translated_query: # Translator now returns original on failure
        print("Proceeding with original query (translation may have failed or wasn't needed).")
        translated_query = query

    # 2. Generate query embedding
    query_embedding = get_query_embedding(translated_query, OPENAI_EMBEDDING_MODEL)
    if query_embedding is None:
        return [] # Cannot search without embedding

    # 3. Build ChromaDB filter
    # IMPORTANT: Assumes domain/vocabulary values in CSV match the case provided here.
    # Consider lowercasing during indexing and query for case-insensitivity.
    filters = []
    if domain:
        filters.append({'domain_id': domain})
        print(f"Filtering by domain_id: {domain}")
    if vocabulary:
        filters.append({'vocabulary_id': vocabulary})
        print(f"Filtering by vocabulary_id: {vocabulary}")

    where_clause = None
    if len(filters) == 1:
        where_clause = filters[0]
    elif len(filters) > 1:
        where_clause = {'$and': filters}

    # 4. Query ChromaDB
    print(f"Querying ChromaDB (k={k})...")
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where_clause, # Apply the constructed filter
            include=['metadatas', 'distances'] # We need distances (scores) and metadata
        )
    except Exception as e:
        print(f"Error during ChromaDB query: {e}")
        return []

    end_time = time.time()
    print(f"Search took {end_time - start_time:.2f} seconds.")

    # 5. Process results
    processed_results = []
    if results and results.get('ids') and results['ids'][0]:
        ids = results['ids'][0]
        distances = results['distances'][0] # Lower distance = more similar for cosine/L2
        metadatas = results['metadatas'][0]

        for concept_id_str, dist, meta in zip(ids, distances, metadatas):
            processed_results.append({
                'concept_id': meta.get('concept_id', 'N/A'), # Get from metadata
                'concept_name': meta.get('concept_name', 'N/A'),
                'domain_id': meta.get('domain_id', 'N/A'),
                'vocabulary_id': meta.get('vocabulary_id', 'N/A'),
                'score': float(dist) # Lower score (distance) is better
            })
        # Results from ChromaDB are already sorted by distance (ascending)
    else:
        print("No results found matching the criteria.")

    return processed_results

def display_results(candidates):
    """Formats and prints the candidate concepts."""
    if not candidates:
        print("\nNo matching standard OMOP concepts found for the specified criteria.")
        return

    print("\nPotential Standard OMOP Concept Matches:")
    print("-" * 70)
    # Results are sorted by distance (lower is better) by ChromaDB query
    for i, concept in enumerate(candidates):
        print(f"{i+1}. Concept ID:   {concept['concept_id']}")
        print(f"   Concept Name: {concept['concept_name']}")
        print(f"   Domain ID:    {concept['domain_id']}")
        print(f"   Vocabulary:   {concept['vocabulary_id']}")
        # Lower score is better (distance)
        print(f"   Score (Distance): {concept['score']:.4f} (Lower score indicates better match)")
        print("-" * 70)

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieve OMOP Concepts using RAG with OpenAI Embeddings.")
    parser.add_argument("query", type=str, help="The input word, phrase, or description (Chinese, Japanese, or English).")
    parser.add_argument("domain", type=str, help="The target OMOP domain to filter by (e.g., Condition, Drug). Case-sensitive match to data.")
    parser.add_argument("-v", "--vocabulary", type=str, default=None, help="(Optional) The target OMOP vocabulary to filter by (e.g., SNOMED, RxNorm). Case-sensitive match to data.")
    parser.add_argument("-k", "--top_k", type=int, default=5, help="Number of results to return.")

    args = parser.parse_args()

    print("\n--- OMOP Concept Retriever (OpenAI + ChromaDB) ---")
    print(f"Query: '{args.query}'")
    print(f"Target Domain: {args.domain}")
    if args.vocabulary:
        print(f"Target Vocabulary: {args.vocabulary}")
    print(f"Number of Results (k): {args.top_k}")
    print("--- Starting Search ---")

    candidates = find_concepts_openai(args.query, args.domain, args.vocabulary, args.top_k)
    display_results(candidates)
    print("--- Search Complete ---")