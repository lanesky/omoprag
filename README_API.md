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

3. Set your API authentication key:
   ```
   # Windows PowerShell
   $env:API_KEY = "your-chosen-api-key"
   
   # Windows Command Prompt
   set API_KEY=your-chosen-api-key
   
   # Linux/Mac
   export API_KEY="your-chosen-api-key"
   ```
   
   Alternatively, you can add this to a `.env` file in the project directory:
   ```
   API_KEY=your-chosen-api-key
   ```
   
   If you don't set an API key, a random one will be generated and displayed when you start the server.

4. Ensure you have built the ChromaDB index for the domains you want to search in. If not, run the vocabulary indexer first.

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

## Authentication

All API endpoints (except `/` and `/health`) require API key authentication. You need to include the API key in the request header:

```
X-API-Key: your-api-key
```

If you don't include a valid API key, the server will respond with a 403 Forbidden error.

## Example Usage with Python Requests

```python
import requests

# Your API key
api_key = "your-api-key"  # The one you set in the environment or .env file
headers = {"X-API-Key": api_key}

# POST request
response = requests.post(
    "http://localhost:8000/concepts/search",
    headers=headers,
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
    headers=headers,
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
curl -X POST "http://localhost:8000/concepts/search" ^
     -H "X-API-Key: your-chosen-api-key" ^
     -H "Content-Type: application/json" ^
     -d "{\"query\": \"風邪\", \"domain\": \"Condition\", \"k\": 5}"

# GET request
curl -X GET "http://localhost:8000/concepts/search?query=%E9%A2%A8%E9%82%AA&domain=Condition&k=5" \
     -H "X-API-Key: your-chosen-api-key"

```
You will get a response like this:

```json
{"query":"風邪","results":[{"concept_id":4202791,"concept_name":"Cold foot","domain_id":"Condition","vocabulary_id":"SNOMED","score":0.4542393684387207},{"concept_id":4152178,"concept_name":"Cold hands","domain_id":"Condition","vocabulary_id":"SNOMED","score":0.4798368215560913},{"concept_id":4154763,"concept_name":"Cold feet","domain_id":"Condition","vocabulary_id":"SNOMED","score":0.4896009564399719},{"concept_id":45765428,"concept_name":"Cold-induced sweating syndrome","domain_id":"Condition","vocabulary_id":"SNOMED","score":0.5000602006912231},{"concept_id":260427,"concept_name":"Common cold","domain_id":"Condition","vocabulary_id":"SNOMED","score":0.5007759928703308}],"count":5,"search_time":1.7644286155700684} 
```


