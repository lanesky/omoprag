# vocabulary_indexer_openai_streaming.py
# (Keep the rest of the script the same, just replace the
# run_openai_streaming_indexing function with this version)

import pandas as pd
import os
import sys
import time
import numpy as np
from openai import OpenAI, RateLimitError, APIError
import chromadb
import gc
import traceback

# --- Assume config_openai import and OpenAI client init are here ---
try:
    from config_openai import (
        OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL, OPENAI_EMBEDDING_BATCH_SIZE,
        CONCEPT_FILE, CHROMA_PATH, COLLECTION_NAME
    )
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    print("OpenAI client initialized successfully.")
except ImportError: print("ERROR: Failed to import configuration from config_openai.py."); exit(1)
except ValueError as e: print(f"ERROR in configuration: {e}"); exit(1)
except Exception as e: print(f"ERROR: Failed to initialize OpenAI client: {e}"); exit(1)

# --- Assume generate_openai_embeddings_batch function is here ---
# (Use the robust version from previous rewrites)
def generate_openai_embeddings_batch(texts: list[str], model: str, batch_size: int, chunk_num_logging: int) -> list | None:
    if not texts: return []
    all_embeddings = []
    num_sub_batches = (len(texts) + batch_size - 1) // batch_size
    print(f"      Generating embeddings for {len(texts)} items in {num_sub_batches} API sub-batches...")
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:min(i + batch_size, len(texts))]
        sub_batch_num = i // batch_size + 1
        print(f"        Processing API sub-batch {sub_batch_num}/{num_sub_batches} ({len(batch_texts)} items)...", end="", flush=True)
        retries = 3; delay = 5; embeddings_obtained = False
        while retries > 0:
            try:
                response = openai_client.embeddings.create(input=batch_texts, model=model)
                batch_embeddings = [item.embedding for item in response.data]
                if len(batch_embeddings) == len(batch_texts):
                    all_embeddings.extend(batch_embeddings); print(" Done."); embeddings_obtained = True; time.sleep(0.6); break
                else: print(f" Mismatch! Expected {len(batch_texts)}, got {len(batch_embeddings)}. Retrying...")
            except RateLimitError: print(f" Rate limit hit. Retrying in {delay}s... ({retries-1})")
            except APIError as e: print(f" API Error ({e.status_code}): {e.message}. Retrying in {delay}s... ({retries-1})")
            except Exception as e: print(f" Unexpected embedding error: {type(e).__name__}. Retrying in {delay}s... ({retries-1})")
            time.sleep(delay); retries -= 1; delay *= 2
        if not embeddings_obtained: print(f"\nERROR: Failed to embed API sub-batch {sub_batch_num} for main chunk {chunk_num_logging} after multiple retries."); return None
    if len(all_embeddings) == len(texts): print(f"      Successfully generated embeddings for all {len(texts)} items in chunk {chunk_num_logging}."); return all_embeddings
    else: print(f"\nERROR: Final embedding count ({len(all_embeddings)}) does not match text count ({len(texts)}) for chunk {chunk_num_logging}."); return None
# --- End of embedding function ---


# --- Main Indexing Logic with Chunk Counting ---
def run_openai_streaming_indexing(domain_id=None):
    """Loads data in chunks, counts chunks first, builds index using OpenAI embeddings,
       adding to ChromaDB in smaller batches."""

    if not os.path.exists(CONCEPT_FILE):
        print(f"ERROR: Concept file not found at {CONCEPT_FILE}"); return
    
    # Dynamically set the ChromaDB path based on domain_id
    chroma_base_path = CHROMA_PATH
    actual_domain = domain_id if domain_id is not None else "All"
    actual_chroma_path = os.path.join(chroma_base_path, actual_domain)
    
    print("--- Starting OpenAI Streaming Indexing (with Chunk Counting) ---")
    print(f"Using Concept File: {CONCEPT_FILE}")
    print(f"Using Domain Filter: {actual_domain}")
    print(f"Using ChromaDB Path: {actual_chroma_path}")
    print(f"Using Collection Name: {COLLECTION_NAME}")
    print(f"Using OpenAI Embedding Model: {OPENAI_EMBEDDING_MODEL}")
    start_total_time = time.time()

    # 1. Setup ChromaDB Client & Collection (explicit handling)
    print("\nInitializing ChromaDB Client...")
    try:
        os.makedirs(actual_chroma_path, exist_ok=True)
        client_chroma = chromadb.PersistentClient(path=actual_chroma_path)
        print(f"ChromaDB client initialized for path: {actual_chroma_path}")
    except Exception as e: print(f"ERROR: Failed to initialize ChromaDB client: {e}"); traceback.print_exc(); return

    collection = None; initial_count = None
    print(f"\nEnsuring ChromaDB collection '{COLLECTION_NAME}' exists...")
    try:
        collection = client_chroma.get_collection(name=COLLECTION_NAME)
        print(f"Successfully retrieved existing collection '{COLLECTION_NAME}'.")
        initial_count = collection.count()
    except Exception as get_err:
        print(f"Collection not found ({get_err}), attempting to create...")
        try:
            collection = client_chroma.create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
            initial_count = 0; print(f"Successfully created new collection '{COLLECTION_NAME}'.")
        except Exception as create_err: print(f"ERROR: Failed to create collection '{COLLECTION_NAME}': {create_err}"); traceback.print_exc(); return
    if collection is None: print("ERROR: Failed to obtain a valid ChromaDB collection. Exiting."); return
    print(f"Collection '{COLLECTION_NAME}' ready. Initial item count: {initial_count}")

    # 2. Define CSV Reading Parameters
    csv_chunk_size = 100000
    db_add_batch_size = 1000
    dtype_spec = { 'concept_id': 'Int64', 'concept_name': str, 'domain_id': str, 'vocabulary_id': str, 'standard_concept': str }
    usecols = list(dtype_spec.keys())

    # 3. **** Count Total Chunks First ****
    print(f"\nPre-counting total chunks in '{os.path.basename(CONCEPT_FILE)}' with chunk size {csv_chunk_size}...")
    total_chunks = 0
    total_rows_counted = 0
    try:
        # Create a temporary iterator just for counting
        counting_iterator = pd.read_csv(
            CONCEPT_FILE, sep='\t', chunksize=csv_chunk_size,
            usecols=['concept_id'], # Read only one column for counting efficiency
            on_bad_lines='skip', # Skip bad lines during count
            quoting=3, encoding='utf-8', low_memory=True
        )
        for count_chunk_df in counting_iterator:
            total_chunks += 1
            total_rows_counted += len(count_chunk_df)
            # Optional: print progress if counting takes long
            # if total_chunks % 10 == 0: print(f"  Counted {total_chunks} chunks...")
        del counting_iterator # Release the iterator object
        del count_chunk_df
        gc.collect()
        if total_chunks == 0 and total_rows_counted > 0:
             total_chunks = 1 # Handle case where file is smaller than chunk size
        print(f"Pre-count complete: Found {total_chunks} total chunks and approx {total_rows_counted} rows.")
        if total_chunks == 0:
            print("WARNING: No data found or readable in the CSV file during pre-count.")
            # Decide if you want to exit here if no chunks are found
            # return
    except Exception as count_err:
        print(f"\nERROR: Failed during CSV pre-counting phase: {count_err}")
        print("Could not determine total number of chunks. Proceeding without total count.")
        traceback.print_exc()
        total_chunks = -1 # Indicate count failed
    # **** End of Chunk Counting ****


    # 4. Process CSV in Chunks (Actual Processing)
    total_processed_rows = 0
    total_added_this_run = 0
    chunk_num = 0
    all_chunks_processed_successfully = True

    print(f"\nStarting processing of {total_chunks if total_chunks > 0 else 'unknown number of'} chunks...")
    print(f"Adding to ChromaDB in batches of {db_add_batch_size} items.")

    csv_iterator = None
    try:
        # Create the main iterator for processing
        csv_iterator = pd.read_csv(
            CONCEPT_FILE, sep='\t', dtype=dtype_spec, usecols=usecols,
            chunksize=csv_chunk_size, on_bad_lines='warn', quoting=3,
            encoding='utf-8', low_memory=False
        )

        for chunk_df in csv_iterator:
            chunk_num += 1
            # Add total chunk count to the log message if available
            chunk_log_prefix = f"--- Processing CSV Chunk {chunk_num}"
            if total_chunks > 0:
                chunk_log_prefix += f"/{total_chunks}"
            chunk_log_prefix += " ---"
            print(f"\n{chunk_log_prefix}")

            start_chunk_time = time.time()
            rows_in_chunk = len(chunk_df)
            total_processed_rows += rows_in_chunk
            print(f"  Read {rows_in_chunk} rows. Total rows processed so far: {total_processed_rows}")

            try:
                # Filter concepts
                valid_standard_mask = ((chunk_df['standard_concept'] == 'S') & chunk_df['concept_name'].notna() & (chunk_df['concept_name'].str.strip() != '') & chunk_df['concept_id'].notna() & (chunk_df['concept_id'] > 0))
                
                # Apply domain_id filter if specified
                if domain_id is not None:
                    valid_standard_mask = valid_standard_mask & (chunk_df['domain_id'] == domain_id)
                    print(f"  Filtering by domain_id: {domain_id}")
                standard_chunk = chunk_df.loc[valid_standard_mask].copy()
                concepts_in_chunk = len(standard_chunk)

                if concepts_in_chunk == 0:
                    print("  No valid standard concepts found in this chunk. Skipping.")
                    continue

                print(f"  Found {concepts_in_chunk} standard concepts for indexing.")

                # Prepare data
                concept_names_full = standard_chunk['concept_name'].astype(str).tolist()
                ids_full = standard_chunk['concept_id'].astype(str).tolist()
                metadata_cols = ['concept_id', 'domain_id', 'vocabulary_id', 'concept_name']
                metadatas_full = standard_chunk[metadata_cols].to_dict('records')
                for meta in metadatas_full: meta['concept_id'] = int(meta['concept_id'])

                # Generate Embeddings
                chunk_embeddings_full = generate_openai_embeddings_batch(
                    texts=concept_names_full, model=OPENAI_EMBEDDING_MODEL,
                    batch_size=OPENAI_EMBEDDING_BATCH_SIZE, chunk_num_logging=chunk_num
                )

                if chunk_embeddings_full is None:
                    print(f"ERROR: Skipping chunk {chunk_num} due to embedding failure.")
                    continue

                # ADD TO DB IN SMALLER BATCHES
                num_db_batches = (concepts_in_chunk + db_add_batch_size - 1) // db_add_batch_size
                print(f"    Adding {concepts_in_chunk} items to ChromaDB in {num_db_batches} batches of size {db_add_batch_size}...")
                chunk_add_success = True
                for i in range(0, concepts_in_chunk, db_add_batch_size):
                    db_batch_start_idx = i; db_batch_end_idx = min(i + db_add_batch_size, concepts_in_chunk)
                    batch_ids = ids_full[db_batch_start_idx:db_batch_end_idx]; batch_embeddings = chunk_embeddings_full[db_batch_start_idx:db_batch_end_idx]
                    batch_documents = concept_names_full[db_batch_start_idx:db_batch_end_idx]; batch_metadatas = metadatas_full[db_batch_start_idx:db_batch_end_idx]
                    db_batch_num = i // db_add_batch_size + 1

                    # ***** START: ADD DEBUG PRINTS *****
                    print(f"\n      --- DEBUG: Preparing DB batch {db_batch_num}/{num_db_batches} ---")
                    print(f"      DEBUG: Number of items in this batch: {len(batch_ids)}")
                    if len(batch_ids) > 0:
                        # Print info for the FIRST item in this specific batch
                        print(f"      DEBUG: First ID:         '{batch_ids[0]}' (Type: {type(batch_ids[0])})")
                        print(f"      DEBUG: First Document:  '{batch_documents[0][:100]}{'...' if len(batch_documents[0])>100 else ''}' (Length: {len(batch_documents[0])})")
                        print(f"      DEBUG: First Metadata:  {batch_metadatas[0]} (Type: {type(batch_metadatas[0])})")
                        if batch_embeddings and len(batch_embeddings) > 0:
                            print(f"      DEBUG: First Embedding Type: {type(batch_embeddings[0])}")
                            if hasattr(batch_embeddings[0], '__len__'):
                                print(f"      DEBUG: First Embedding Dim: {len(batch_embeddings[0])}")
                                if len(batch_embeddings[0]) > 5:
                                    print(f"      DEBUG: First Embedding Start: {batch_embeddings[0][:5]}")
                            else:
                                 print(f"      DEBUG: First Embedding is not a sequence.")
                        else:
                            print(f"      DEBUG: Embeddings list is empty or invalid for first item.")
                        # Check types for the whole batch (optional but useful)
                        all_ids_str = all(isinstance(item, str) for item in batch_ids)
                        all_embeds_list = all(isinstance(item, list) for item in batch_embeddings)
                        all_metas_dict = all(isinstance(item, dict) for item in batch_metadatas)
                        print(f"      DEBUG: All IDs strings? {all_ids_str}. All Embeddings lists? {all_embeds_list}. All Metadatas dicts? {all_metas_dict}.")
                    else:
                         print("      DEBUG: This DB batch is empty.")
                    print(f"      --- END DEBUG ---")
                    # ***** END: ADD DEBUG PRINTS *****

                    print(f"      Adding DB batch {db_batch_num}/{num_db_batches} ({len(batch_ids)} items)...", end="", flush=True)
                    start_add_time = time.time()
                    try:
                        if collection is None: print("\nERROR: Collection invalid."); chunk_add_success = False; break
                        collection.add(ids=batch_ids, embeddings=batch_embeddings, documents=batch_documents, metadatas=batch_metadatas)
                        total_added_this_run += len(batch_ids); end_add_time = time.time(); print(f" Done in {end_add_time - start_add_time:.2f}s."); time.sleep(0.2)
                    except Exception as db_add_error:
                        print(f"\nERROR: Failed to add DB batch {db_batch_num} for chunk {chunk_num}: {db_add_error}"); traceback.print_exc(limit=2)
                        print(f"      Skipping remaining DB batches for chunk {chunk_num}."); chunk_add_success = False; break

                if chunk_add_success: print(f"    Successfully added all DB batches for chunk {chunk_num}.")

            except Exception as chunk_processing_error:
                print(f"\nERROR: Unexpected error processing chunk {chunk_num}: {chunk_processing_error}"); traceback.print_exc(); print(f"Attempting to continue with next chunk...")

            finally:
                # Memory Management
                try: del chunk_df; del standard_chunk; del concept_names_full; del ids_full; del metadatas_full; del chunk_embeddings_full; del batch_ids; del batch_embeddings; del batch_documents; del batch_metadatas
                except NameError: pass
                if chunk_num % 5 == 0: gc.collect()
                end_chunk_time = time.time()
                # Add total chunk count to the log message if available
                finish_log_prefix = f"--- Finished processing CSV Chunk {chunk_num}"
                if total_chunks > 0:
                     finish_log_prefix += f"/{total_chunks}"
                finish_log_prefix += f" in {end_chunk_time - start_chunk_time:.2f} seconds ---"
                print(finish_log_prefix)


    except FileNotFoundError: print(f"ERROR: File {CONCEPT_FILE} not found."); all_chunks_processed_successfully = False
    except pd.errors.ParserError as pe: print(f"\nERROR: Failed to parse CSV file near row ~{total_processed_rows}. Details: {pe}"); all_chunks_processed_successfully = False; traceback.print_exc(limit=5)
    except Exception as e: print(f"\nERROR: Unexpected error during CSV reading/iteration after chunk {chunk_num}: {e}"); all_chunks_processed_successfully = False; traceback.print_exc()

    finally:
        # Final Summary
        # ... (Summary code remains the same) ...
        end_total_time = time.time()
        print("\n--- Indexing Process Summary ---")
        if csv_iterator is None: print("CSV iterator could not be initialized.")
        elif all_chunks_processed_successfully and (total_chunks == -1 or chunk_num == total_chunks): print(f"Successfully processed all {chunk_num} chunks.") # Adjusted condition
        else: print(f"WARNING: Processing may have stopped prematurely after chunk {chunk_num}/{total_chunks if total_chunks > 0 else '?'} due to errors.")
        print(f"Total rows read from CSV: {total_processed_rows}")
        print(f"Total standard concepts added/updated in this run: {total_added_this_run}")
        try:
            if collection: final_count = collection.count(); print(f"Final ChromaDB collection ('{COLLECTION_NAME}') count: {final_count}")
            else: print("Could not get final collection count as collection object is invalid.")
        except Exception as e: print(f"Could not get final collection count: {e}")
        print(f"Total indexing time: {end_total_time - start_total_time:.2f} seconds.")
        print("--- Streaming Indexing Complete ---")


if __name__ == "__main__":
    domain_id = None
    # Check if domain_id is provided as command line argument
    if len(sys.argv) > 1:
        domain_id = sys.argv[1]
        print(f"Using domain_id filter: {domain_id}")
    run_openai_streaming_indexing(domain_id)