FROM python:3.9-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y git gcc

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download models (cached in Docker)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
RUN python -c "import spacy; spacy.cli.download('en_core_web_sm')"

# Copy app code
COPY . .

CMD ["python", "app.py"]