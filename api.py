"""
OMOP Concept Retrieval API
This module provides a FastAPI implementation for the OMOP concept retrieval functionality.
It exposes the concept retrieval logic from concept_retriever_openai_quick.py as REST API endpoints.
"""

import os
import time
import secrets
from typing import List, Optional, Dict, Any, Union
from fastapi import FastAPI, HTTPException, Query, Depends, Header, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader, APIKey
from pydantic import BaseModel, Field
import uvicorn
from dotenv import load_dotenv

# Import the concept retrieval functionality
try:
    from concept_retriever_openai_quick import find_concepts_openai, display_results
except ImportError:
    raise ImportError("Failed to import from concept_retriever_openai_quick.py. Make sure the file exists and is accessible.")

# Load environment variables
load_dotenv()

# API key settings
API_KEY_NAME = "X-API-Key"
API_KEY = os.getenv("API_KEY") or secrets.token_urlsafe(32)  # Generate a random key if not set

if not os.getenv("API_KEY"):
    print(f"WARNING: API_KEY environment variable not set. Using generated key: {API_KEY}")
    print("Set this key in your environment or .env file for persistent authentication.")

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Create FastAPI app
app = FastAPI(
    title="OMOP Concept Retrieval API",
    description="API for retrieving OMOP standard concepts using OpenAI embeddings",
    version="1.0.0"
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Define Pydantic models for request and response
class ConceptRequest(BaseModel):
    query: str = Field(..., description="The query text to search for concepts")
    domain: str = Field(..., description="The domain to search in (e.g., 'Condition', 'Drug', 'Procedure')")
    vocabulary: Optional[str] = Field(None, description="Optional vocabulary filter")
    k: int = Field(5, description="Number of results to return", ge=1, le=100)
    db_domain: Optional[str] = Field(None, description="Optional database domain if different from search domain")

class ConceptResponse(BaseModel):
    concept_id: Union[int, str] = Field(..., description="OMOP concept ID")
    concept_name: str = Field(..., description="OMOP concept name")
    domain_id: str = Field(..., description="Domain ID")
    vocabulary_id: str = Field(..., description="Vocabulary ID")
    score: float = Field(..., description="Similarity score")

class ConceptSearchResponse(BaseModel):
    query: str = Field(..., description="Original query")
    results: List[ConceptResponse] = Field(..., description="List of matching concepts")
    count: int = Field(..., description="Number of results returned")
    search_time: float = Field(..., description="Search time in seconds")

# API key dependency
async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid API Key",
        )
    return api_key

# Define API endpoints
@app.get("/", tags=["Info"])
async def root():
    """Root endpoint that provides basic API information."""
    return {
        "message": "OMOP Concept Retrieval API",
        "version": "1.0.0",
        "endpoints": {
            "/concepts/search": "Search for OMOP concepts",
            "/health": "Health check endpoint"
        }
    }

@app.get("/health", tags=["Info"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": time.time()}

@app.post("/concepts/search", response_model=ConceptSearchResponse, tags=["Concepts"])
async def search_concepts(request: ConceptRequest, api_key: APIKey = Depends(get_api_key)):
    """
    Search for OMOP concepts based on the provided query.
    
    This endpoint uses OpenAI embeddings to find semantically similar concepts
    in the OMOP vocabulary.
    """
    start_time = time.time()
    
    try:
        # Call the existing function from concept_retriever_openai_quick.py
        results = find_concepts_openai(
            query=request.query,
            domain=request.domain,
            vocabulary=request.vocabulary,
            k=request.k,
            db_domain=request.db_domain
        )
        
        # Convert results to the response format
        response_results = []
        for concept in results:
            response_results.append(
                ConceptResponse(
                    concept_id=concept['concept_id'],
                    concept_name=concept['concept_name'],
                    domain_id=concept['domain_id'],
                    vocabulary_id=concept['vocabulary_id'],
                    score=concept['score']
                )
            )
        
        # Create and return the response
        return ConceptSearchResponse(
            query=request.query,
            results=response_results,
            count=len(response_results),
            search_time=time.time() - start_time
        )
    
    except Exception as e:
        # Log the error and raise an HTTP exception
        error_message = f"Error searching for concepts: {str(e)}"
        print(error_message)
        raise HTTPException(status_code=500, detail=error_message)

# GET endpoint alternative for simple queries
@app.get("/concepts/search", response_model=ConceptSearchResponse, tags=["Concepts"])
async def search_concepts_get(
    query: str = Query(..., description="The query text to search for concepts"),
    domain: str = Query(..., description="The domain to search in (e.g., 'Condition', 'Drug', 'Procedure')"),
    vocabulary: Optional[str] = Query(None, description="Optional vocabulary filter"),
    k: int = Query(5, description="Number of results to return", ge=1, le=100),
    db_domain: Optional[str] = Query(None, description="Optional database domain if different from search domain"),
    api_key: APIKey = Depends(get_api_key)
):
    """
    Search for OMOP concepts based on the provided query parameters.
    
    This is a GET alternative to the POST endpoint, useful for simple queries
    or when using the API directly from a browser.
    """
    # Create a request object and call the POST endpoint handler
    request = ConceptRequest(
        query=query,
        domain=domain,
        vocabulary=vocabulary,
        k=k,
        db_domain=db_domain
    )
    return await search_concepts(request)

# Run the API server when the script is executed directly
if __name__ == "__main__":
    # You can change host and port as needed
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
