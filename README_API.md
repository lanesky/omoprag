# OMOP Concept Retrieval API

This API provides a FastAPI implementation for retrieving OMOP standard concepts using OpenAI embeddings.

## Setup

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Make sure your OpenAI API key is set in the environment:
   ```
   # Windows PowerShell
   $env:OPENAI_API_KEY = "your-api-key"
   
   # Windows Command Prompt
   set OPENAI_API_KEY=your-api-key
   
   # Linux/Mac
   export OPENAI_API_KEY="your-api-key"
   ```

3. Ensure you have built the ChromaDB index for the domains you want to search in. If not, run the vocabulary indexer first.

## Running the API

Start the API server:

```
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at http://localhost:8000

## API Documentation

Once the server is running, you can access the auto-generated Swagger documentation at:

http://localhost:8000/docs

This provides an interactive interface to test the API endpoints.

## Endpoints

### GET /

Returns basic information about the API.

### GET /health

Health check endpoint.

### POST /concepts/search

Search for OMOP concepts based on the provided query.

**Request Body:**
```json
{
  "query": "heart attack",
  "domain": "Condition",
  "vocabulary": null,
  "k": 5,
  "db_domain": null
}
```

### GET /concepts/search

Alternative GET endpoint for simple queries.

**Query Parameters:**
- `query`: The query text to search for concepts
- `domain`: The domain to search in (e.g., 'Condition', 'Drug', 'Procedure')
- `vocabulary` (optional): Vocabulary filter
- `k` (optional, default=5): Number of results to return
- `db_domain` (optional): Database domain if different from search domain

**Example:**
```
GET /concepts/search?query=heart%20attack&domain=Condition&k=10
```

## Example Usage with Python Requests

```python
import requests

# POST request
response = requests.post(
    "http://localhost:8000/concepts/search",
    json={
        "query": "heart attack",
        "domain": "Condition",
        "k": 5
    }
)
results = response.json()
print(results)

# GET request
response = requests.get(
    "http://localhost:8000/concepts/search",
    params={
        "query": "heart attack",
        "domain": "Condition",
        "k": 5
    }
)
results = response.json()
print(results)
```

## Example Usage with cURL

```bash
# POST request
curl -X POST "http://localhost:8000/concepts/search" \
     -H "Content-Type: application/json" \
     -d '{"query":"heart attack","domain":"Condition","k":5}'

# GET request
curl -X GET "http://localhost:8000/concepts/search?query=heart%20attack&domain=Condition&k=5"
```
