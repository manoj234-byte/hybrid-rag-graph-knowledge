import spacy
import networkx as nx
import pinecone
from data_processor import load_embedding_model, get_pinecone_index
from kg_builder import load_knowledge_graph

# Load SciSpacy model for biomedical NER
try:
    nlp_med = spacy.load("en_core_sci_sm")
except OSError:
    print("SciSpacy model 'en_core_sci_sm' not found. Falling back to 'en_core_web_sm'.")
    nlp_med = spacy.load("en_core_web_sm")

class HybridRetriever:
    def __init__(self):
        self.embedding_model = load_embedding_model()
        self.pinecone_index = get_pinecone_index()
        self.kg = load_knowledge_graph()
        self.kg_nlp = nlp_med

        self.synonyms = {
            "high blood sugar": ["hyperglycemia", "diabetes"],
            "heart attack": ["myocardial infarction", "mi"],
            "high blood pressure": ["hypertension"],
            "flu": ["influenza"],
            "covid-19": ["coronavirus", "sars-cov-2"]
        }

    def retrieve_vector(self, query: str, top_k: int = 3) -> list[str]:
        query_embedding = self.embedding_model.encode(query).tolist()
        results = self.pinecone_index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace="documents"
        )
        return [m.metadata.get('text', '') for m in results.matches]

    def retrieve_graph(self, query: str, max_hops: int = 1) -> str:
        doc = self.kg_nlp(query)
        entities = {ent.text.lower() for ent in doc.ents}
        entities |= {tok.text.lower() for tok in doc if tok.pos_ in ["NOUN", "PROPN"]}

        expanded = set()
        for ent in entities:
            expanded.add(ent)
            if ent in self.synonyms:
                expanded.update(self.synonyms[ent])
        entities = expanded

        context = []
        valid_relations = {"has_symptom", "symptom", "exhibits", "causes", "treated_by", "treated_with", "involves", "indicates"}

        for ent in entities:
            if ent in self.kg:
                for nbr in self.kg.neighbors(ent):
                    ed = self.kg.get_edge_data(ent, nbr)
                    for _, data in ed.items():
                        relation = data.get('relation') if isinstance(data, dict) else str(data)
                        if relation.lower() in valid_relations:
                            context.append(f"- {ent.capitalize()} {relation} {nbr.capitalize()}")

                for nbr in self.kg.predecessors(ent):
                    ed = self.kg.get_edge_data(nbr, ent)
                    for _, data in ed.items():
                        relation = data.get('relation') if isinstance(data, dict) else str(data)
                        if relation.lower() in valid_relations:
                            context.append(f"- {nbr.capitalize()} {relation} {ent.capitalize()}")

        return "\n".join(context) if context else "No direct graph relations found for query entities."

    def hybrid_retrieve(self, query: str, vector_top_k: int = 3, graph_max_hops: int = 1):
        docs = self.retrieve_vector(query, vector_top_k)
        kg_ctx = self.retrieve_graph(query, graph_max_hops)
        return docs, kg_ctx
