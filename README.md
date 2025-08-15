---
title: Hybrid RAG with Graph Knowledge
emoji: 😻
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---
# Hybrid RAG with Graph Knowledge Integration

## Project Overview

This project implements an advanced Retrieval-Augmented Generation (RAG) system that combines traditional vector-based search with a simplified graph-based knowledge representation. The goal is to provide more contextually aware and relationship-rich responses by leveraging both the semantic similarity of text embeddings and the structured nature of a knowledge graph.

## Features

*   **Traditional Vector Search**: Utilizes Hugging Face `sentence-transformers` for embeddings and Pinecone as the vector database for efficient semantic search.
*   **Graph-Based Knowledge Representation**: Extracts entities and simple relations from text using `spaCy` and stores them in an in-memory `networkx` graph.
*   **Hybrid Retrieval**: Combines results from vector search (relevant text chunks) and graph traversal (related entities/facts) to create a richer context.
*   **Context-Aware Generation**: Employs a Hugging Face causal language model (`google/flan-t5-base`) to synthesize answers based on the combined retrieved context.
*   **User-Friendly Interface**: Built with Streamlit for an interactive and accessible web demo.

## Technical Stack

*   **Embedding Model**: `sentence-transformers` (`all-MiniLM-L6-v2`)
*   **Vector Database**: Pinecone
*   **Knowledge Graph**: `spaCy` for NLP, `networkx` for graph representation
*   **Language Model**: Hugging Face Transformers (`google/flan-t5-base`)
*   **Application Framework**: Streamlit
*   **Environment Management**: `python-dotenv` for local secrets
*   **Deployment**: Docker (for Hugging Face Spaces)

## Local Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd hybrid_rag_project
    ```
    (Replace `<your-repo-url>` with your actual GitHub repository URL once you set it up.)

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows: `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download spaCy model:**
    The knowledge graph builder relies on spaCy.
    ```bash
    python -m spacy download en_core_web_sm
    ```

5.  **Set up Pinecone API Key:**
    *   Go to [Pinecone](https://www.pinecone.io/) and sign up for a free account.
    *   Obtain your API Key and Environment (e.g., `us-west-2`).
    *   Create a file named `.env` in the root of your `hybrid_rag_project` directory and add:
        ```
        PINECONE_API_KEY="YOUR_API_KEY"
        PINECONE_ENVIRONMENT="YOUR_ENVIRONMENT"
        ```
        Replace `"YOUR_API_KEY"` and `"YOUR_ENVIRONMENT"` with your actual credentials.

## Running the Project Locally

1.  **Prepare Data and Knowledge Graph:**
    The first time you run the Streamlit app, it will attempt to process the `sample_data.txt`, embed it into Pinecone, and build the knowledge graph. This can take a few minutes. You can also manually run these steps once:
    ```bash
    python src/data_processor.py
    python src/kg_builder.py
    ```
    This will create your Pinecone index and the `data/knowledge_graph.json` file.

2.  **Start the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
    This will open the application in your default web browser (usually `http://localhost:8501`).

## Usage

1.  Enter a query related to the sample data (e.g., "What is Cancer?", "Tell me about the COVID.", "How find COVID?").
2.  Click "Get Answer".
3.  The system will display the retrieved text chunks (from Pinecone), relevant facts from the knowledge graph, and the final synthesized answer from the LLM.

## Knowledge Graph Limitations & Simplifications

*   The knowledge graph construction in `kg_builder.py` uses a **simplified rule-based approach** with `spaCy`. It primarily extracts basic subject-verb-object patterns and named entities. For a production system, a more robust information extraction pipeline (e.g., using pre-trained relation extraction models, coreference resolution, and entity linking against external knowledge bases like Wikidata) would be required.
*   The `networkx` graph is **in-memory**. While sufficient for this demo with small data, for very large knowledge bases, a dedicated graph database (like Neo4j) would be necessary.
*   **Entity Linking**: Simple lowercasing is used for entity normalization. Advanced entity linking is not implemented, meaning "AI" and "Artificial Intelligence" might not always resolve to the same node without explicit mapping.

## Future Enhancements

*   Implement a more sophisticated Knowledge Graph extraction pipeline.
*   Integrate fuzzy entity linking or use a pre-built entity linker.
*   Explore advanced hybrid retrieval strategies (e.g., re-ranking retrieved documents based on graph relevance, generating graph-aware queries for the LLM).
*   Add more comprehensive evaluation metrics (e.g., RAGAS) and a test dataset.
*   Improve the UX/UI with more detailed context visualization.
*   Support for larger language models (requires more powerful compute resources).
*   Add a mechanism to easily update the Pinecone index and knowledge graph with new data.


---
