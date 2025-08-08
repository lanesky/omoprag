# OMOP Concept Retriever with OpenAI and ChromaDB

This project provides a set of tools to build a semantic search engine for OMOP concepts using OpenAI's language models and ChromaDB for vector storage. It allows you to build an index of OMOP concepts and then retrieve them based on natural language queries.

## Features

-   **Concept Indexing**: Build a vector index of OMOP concepts from a `CONCEPT.csv` file.
-   **Semantic Search**: Retrieve OMOP concepts using natural language queries in English, Chinese, or Japanese.
-   **Domain Filtering**: Filter concepts by specific domains (e.g., Drug, Condition).
-   **FastAPI**: Expose the concept retrieval functionality as a REST API.
-   **Fly.io Deployment**: Includes instructions for deploying the application using fly.io.

## Project File Overview

-   `config_openai.py`: Contains all project configuration parameters.
-   `vocabulary_indexer_openai_streaming.py`: A streaming index builder for large datasets.
-   `concept_retriever_openai.py`: The main tool for retrieving concepts.
-   `concept_retriever_openai_quick.py`: An optimized version of the concept retriever.
-   `translator_openai.py`: A utility for translating medical terms.
-   `api.py`: A FastAPI implementation for the concept retrieval API.

## Setup and Installation

1.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Set environment variables**:
    You need to set your OpenAI API key. You can do this by setting an environment variable or by creating a `.env` file in the project root.

    *   **Environment variable**:
        ```bash
        # Linux/Mac
        export OPENAI_API_KEY="your-api-key"

        # Windows PowerShell
        $env:OPENAI_API_KEY = "your-api-key"
        ```
    *   **.env file**:
        Create a file named `.env` in the project root and add the following line:
        ```
        OPENAI_API_KEY=your-api-key
        ```

## Data Preparation and Indexing

1.  **Obtain OMOP Concept Data**:
    *   Download the OMOP Vocabulary data from [ATHENA](https://athena.ohdsi.org/).
    *   Extract the `CONCEPT.csv` file.

2.  **Prepare Index Data**:
    *   Filter for standard concepts from the `CONCEPT.csv` file.
    *   Place the filtered `CONCEPT.csv` file in the `csv_for_indexing` directory.

3.  **Build the Index**:
    *   To build an index for all concepts:
        ```bash
        python vocabulary_indexer_openai_streaming.py
        ```
    *   To build an index for a specific domain:
        ```bash
        python vocabulary_indexer_openai_streaming.py Drug
        ```

## Usage

Once the index is built, you can retrieve concepts using the command-line tools.

-   **Search in a specific domain**:
    ```bash
    python concept_retriever_openai.py "hypertension" Condition
    ```

-   **Search in all domains**:
    ```bash
    python concept_retriever_openai.py "hypertension" Condition --use-all-db
    ```

-   **Use the optimized retriever**:
    ```bash
    python concept_retriever_openai_quick.py "hypertension" Condition
    ```

## API

The project includes a FastAPI for serving the concept retrieval functionality.

1.  **Set API Key**:
    Set an `API_KEY` for the API in your environment variables or `.env` file. If you don't set one, a random key will be generated.
    ```bash
    export API_KEY="your-chosen-api-key"
    ```

2.  **Run the API**:
    ```bash
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
    ```

3.  **API Documentation**:
    Once the server is running, you can access the interactive Swagger documentation at `http://localhost:8000/docs`.

4.  **Example API usage**:
    *   **cURL**:
        ```bash
        curl -X POST "http://localhost:8000/concepts/search" \
             -H "X-API-Key: your-chosen-api-key" \
             -H "Content-Type: application/json" \
             -d '{"query": "heart attack", "domain": "Condition"}'
        ```
    *   **Python**:
        ```python
        import requests

        api_key = "your-chosen-api-key"
        headers = {"X-API-Key": api_key}

        response = requests.post(
            "http://localhost:8000/concepts/search",
            headers=headers,
            json={
                "query": "heart attack",
                "domain": "Condition",
                "k": 5
            }
        )
        print(response.json())
        ```

## Deployment (fly.io)

You can deploy this application using [fly.io](https://fly.io/).

-   **Deploy**:
    ```bash
    fly deploy
    ```

-   **Set secrets**:
    ```bash
    fly secrets set OPENAI_API_KEY="your-openai-api-key"
    fly secrets set API_KEY="your-chosen-api-key"
    ```

-   **SSH access**:
    ```bash
    fly ssh console
    ```

-   **Manage volumes**:
    ```bash
    fly volumes list
    ```
