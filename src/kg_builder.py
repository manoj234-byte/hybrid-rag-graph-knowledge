import spacy
import networkx as nx
import os
import json
from tqdm.auto import tqdm

try:
    nlp = spacy.load("en_core_sci_sm")
except OSError:
    print("SciSpacy model not found—falling back to en_core_web_sm.")
    nlp = spacy.load("en_core_web_sm")

KG_PATH = "data/health_knowledge_graph.json"

def normalize(text: str) -> str:
    return text.lower().strip().replace(".", "").replace(",", "")

def extract_entities_and_relations(text: str):
    doc = nlp(text)
    triples = []

    for sent in doc.sents:
        s_doc = nlp(sent.text)

        for token in s_doc:
            if token.pos_ == "VERB":
                subj = [w for w in token.lefts if w.dep_ in ("nsubj", "nsubjpass")]
                obj = [w for w in token.rights if w.dep_ in ("dobj", "attr", "pobj")]

                if subj and obj:
                    subj_text = " ".join(w.text for w in subj).lower()
                    obj_text = " ".join(w.text for w in obj).lower()
                    triples.append((subj_text, token.lemma_.lower(), obj_text))

        for ent in s_doc.ents:
            triples.append((ent.text.lower(), "is_a", ent.label_.lower()))

    return triples

def build_knowledge_graph(data_path: str = "data/health_corpus.txt"):
    G = nx.DiGraph()

    if not os.path.exists(data_path):
        print(f"Health corpus not found at {data_path}.")
        return

    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()

    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    all_triples = []
    for para in tqdm(paragraphs, desc="Extracting triples"):
        all_triples.extend(extract_entities_and_relations(para))

    for s, p, o in tqdm(all_triples, desc="Building graph"):
        s_norm, o_norm = normalize(s), normalize(o)
        relation = p.lower().replace("has symptom", "has_symptom") \
                            .replace("treated by", "treated_by") \
                            .replace("treated with", "treated_with")
        if s_norm and o_norm and s_norm != o_norm:
            G.add_node(s_norm)
            G.add_node(o_norm)
            G.add_edge(s_norm, o_norm, relation=relation)

    os.makedirs(os.path.dirname(KG_PATH), exist_ok=True)
    with open(KG_PATH, 'w', encoding='utf-8') as f:
        json.dump(nx.node_link_data(G), f, indent=2)

    print(f"Built KG with {G.number_of_nodes()} nodes, {G.number_of_edges()} edges. Saved to {KG_PATH}.")

def load_knowledge_graph():
    if not os.path.exists(KG_PATH):
        raise FileNotFoundError(f"KG file not found: {KG_PATH}")
    with open(KG_PATH, 'r', encoding='utf-8') as f:
        return nx.node_link_graph(json.load(f))

if __name__ == "__main__":
    build_knowledge_graph()
