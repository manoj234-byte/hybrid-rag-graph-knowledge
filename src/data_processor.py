# src/data_processor.py
import os
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from utils import get_or_create_pinecone_index, INDEX_NAME  
from tqdm.auto import tqdm


NLTK_DATA_PATH = os.environ.get("NLTK_DATA", "/app/nltk_data")
nltk.data.path.append(NLTK_DATA_PATH)


try:
    nltk.data.find('tokenizers/punkt')
except LookupError:  # <-- Correct exception type here
    nltk.download('punkt', download_dir=NLTK_DATA_PATH)


embedding_model = None
pinecone_index = None
EMBEDDING_DIM = 384  # Dimension for 'all-MiniLM-L6-v2'

def load_embedding_model():
    """Loads the Sentence Transformer model."""
    global embedding_model
    if embedding_model is None:
        print("Loading Sentence Transformer model 'all-MiniLM-L6-v2'...")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Model loaded.")
    return embedding_model

def get_pinecone_index():
    """Initializes and returns the Pinecone index."""
    global pinecone_index
    if pinecone_index is None:
        pinecone_index = get_or_create_pinecone_index(EMBEDDING_DIM)
    return pinecone_index

def chunk_text(text: str, max_chunk_size: int = 250) -> list[str]:
    """Chunks text into sentences and then combines them into larger chunks."""
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split()) 
        if current_length + sentence_length <= max_chunk_size:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def process_and_upsert_data(data_path: str = "data/health_corpus.txt"):
    """
    Reads data, chunks it, generates embeddings, and upserts to Pinecone.
    """
    model = load_embedding_model()
    index = get_pinecone_index()

    with open(data_path, 'r', encoding='utf-8') as f:
        full_text = f.read()

    chunks = chunk_text(full_text)
    print(f"Generated {len(chunks)} chunks.")

    
    vectors_to_upsert = []
    for i, chunk in enumerate(tqdm(chunks, desc="Generating embeddings and preparing for upsert")):
        embedding = model.encode(chunk).tolist()
        vectors_to_upsert.append({
            "id": f"chunk-{i}",
            "values": embedding,
            "metadata": {"text": chunk, "source": data_path}
        }))
    batch_size = 100
    for i in tqdm(range(0, len(vectors_to_upsert), batch_size), desc="Upserting to Pinecone"):
        batch = vectors_to_upsert[i:i + batch_size]
        index.upsert(vectors=batch, namespace="documents")  # Use a namespace for organization
    print(f"Successfully upserted {len(vectors_to_upsert)} vectors to Pinecone.")

if __name__ == "__main__":
    
    os.makedirs('data', exist_ok=True)
    process_and_upsert_data()
    
