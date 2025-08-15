from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
from typing import List

text_generation_pipeline = None

def load_llm_pipeline():
    global text_generation_pipeline
    if text_generation_pipeline is None:
        model_name = "google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        text_generation_pipeline = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
    return text_generation_pipeline

def truncate(text: str, max_tokens: int, tokenizer):
    tokens = tokenizer.tokenize(text)
    if len(tokens) > max_tokens:
        return tokenizer.convert_tokens_to_string(tokens[:max_tokens])
    return text

def generate_response(query: str, retrieved_docs: List[str], graph_context: List[str]) -> str:
    pipeline_llm = load_llm_pipeline()
    tokenizer = pipeline_llm.tokenizer

    context_text = "\n".join(retrieved_docs)
    context_text = truncate(context_text, 300, tokenizer)

    graph_text = "\n".join(f"- {item}" for item in graph_context) if graph_context else ""
    graph_text = truncate(graph_text, 100, tokenizer) if graph_text else "No relevant medical knowledge found."

    prompt = (
        f"You are a medical expert. Use the info below to answer clearly.\n\n"
        f"Documents:\n{context_text}\n\n"
        f"Knowledge Graph:\n{graph_text}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )

    outputs = pipeline_llm(
        prompt,
        max_length=150,
        do_sample=True,
        top_p=0.95,
        temperature=0.7,
        num_return_sequences=1
    )
    text = outputs[0]['generated_text']
    answer = text.split("Answer:")[-1].strip() if "Answer:" in text else text.strip()

    if not answer or len(answer.split()) < 3:
        return "Symptoms may include fatigue, nausea, and other physiological changes. Consult a healthcare provider for a full diagnosis."
    return answer
