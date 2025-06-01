import chromadb
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(path="./data/chroma_db")
collection = client.get_or_create_collection(name="ehr_data")

# Mock EHR data
documents = [
    "Patient John Doe, HbA1c: 6.5%, LDL: 110 mg/dL (Last visit: 2024-05-10)",
    "Patient Jane Smith, HbA1c: 5.8%, LDL: 95 mg/dL (Last visit: 2024-05-12)",
]
embeddings = SentenceTransformer('all-MiniLM-L6-v2').encode(documents).tolist()

collection.add(
    embeddings=embeddings,
    documents=documents,
    ids=[f"id_{i}" for i in range(len(documents))]
)