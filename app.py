from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI()

# Load models
logger.info("Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")  # ~80MB

logger.info("Loading Phi-3-mini LLM...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-3-mini-4k-instruct")
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-3-mini-4k-instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Initialize ChromaDB
logger.info("Initializing ChromaDB...")
client = chromadb.PersistentClient(path="./data/chroma_db")
collection = client.get_or_create_collection(name="ehr_data")

# Request model
class Query(BaseModel):
    text: str

# Health check
@app.get("/ping")
def ping():
    return {"status": "alive"}

# RAG + LLM endpoint
@app.post("/ask")
def ask_ehr(query: Query):
    logger.info(f"Received query: {query.text}")
    
    try:
        # 1. Retrieve relevant EHR data
        query_embedding = embed_model.encode(query.text).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )
        context = "\n".join(results["documents"][0]) if results["documents"] else "No relevant EHR data found"
        logger.info(f"Retrieved context: {context[:100]}...")  # Log first 100 chars

        # 2. Generate response with Phi-3
        messages = [
            {"role": "system", "content": "You are a medical assistant. Answer using ONLY the provided EHR data. Be concise."},
            {"role": "user", "content": f"EHR Data:\n{context}\n\nQuestion: {query.text}"}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        logger.debug(f"Full prompt:\n{prompt}")

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False
        )
        
        # 3. Clean response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from the output if it is echoed
        if prompt in full_response:
            response = full_response.replace(prompt, "").strip()
        else:
            response = full_response.strip()
        logger.info(f"Generated response: {response}")

        return {"response": response}

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001, log_level="info")