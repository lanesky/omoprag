FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV CHROMA_PATH=/data/chroma_db
ENV PORT=8080

# Expose the port
EXPOSE 8080

# Command to run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
