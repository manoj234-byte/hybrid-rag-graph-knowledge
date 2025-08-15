import streamlit as st
import os
from retriever import HybridRetriever
from generator import generate_response
from data_processor import process_and_upsert_data
from kg_builder import build_knowledge_graph

st.set_page_config(page_title="Health Hybrid RAG", layout="wide")
st.title("Hybrid RAG with Graph KG (Health Domain)")

st.markdown(
    "This demo combines vector search and a medical knowledge graph to deliver context-aware "
    "health responses."
)

@st.cache_resource(show_spinner=False)
def initialize_system():
    st.info("Initializing Health RAG system...")
    os.makedirs("data", exist_ok=True)
    process_and_upsert_data(data_path="data/health_corpus.txt")
    build_knowledge_graph(data_path="data/health_corpus.txt")
    st.success("Ready! Knowlege Graph & Vector Index initialized.")
    return HybridRetriever()

try:
    retriever = initialize_system()
except Exception as e:
    st.error("Initialization failed.")
    st.exception(e)
    st.stop()

query = st.text_input("Ask a medical question:", "What are the symptoms and treatments of diabetes?")

if st.button("Get Answer"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Retrieving..."):
            docs, kg_ctx = retriever.hybrid_retrieve(query)
        st.subheader("📄 Document Context")
        for i, doc in enumerate(docs, 1):
            st.write(f"**Chunk {i}:** {doc[:200]}{'...' if len(doc) > 200 else ''}")

        st.subheader("🩺 Knowledge Graph Context")
        if "No direct" in kg_ctx:
            st.info("No relevant relations found in the Knowledge Graph.")
        else:
            st.code(kg_ctx)

        answer = generate_response(query, docs, kg_ctx.splitlines())
        st.subheader("💡 Answer")
        st.write(answer)

st.sidebar.header("How It Works")
st.sidebar.write(
    "1. **Vector Search** retrieves relevant health text.\n"
    "2. **KG Search** surfaces medical relations via graph.\n"
    "3. **LLM** (Flan-T5) generates a detailed, accurate response."
)
