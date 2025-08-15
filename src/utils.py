import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "hybrid-rag-index"

def init_pinecone():
    """Initializes Pinecone client and returns it."""
    if not PINECONE_API_KEY:
        raise ValueError("Pinecone API key not set in .env file.")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc

def get_or_create_pinecone_index(dimension: int):
    """Gets or creates a Pinecone index, returns the Index object."""
    pc = init_pinecone()

    spec = ServerlessSpec(
        cloud="aws",        
        region="us-east-1"  
    )

 
    if INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating Pinecone index '{INDEX_NAME}' with dimension {dimension}...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=dimension,
            metric='cosine',
            spec=spec
        )
        print("Index created.")
    else:
        print(f"Pinecone index '{INDEX_NAME}' already exists.")

    return pc.Index(INDEX_NAME)
