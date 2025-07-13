from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import os

MODEL_PATH = "models/lora_trex"
DATA_PATH = "data/corpus_ft/preprocessed_tfidf.pkl"
OUT_PATH = "data/triples_raw/triples_bbc.pkl"
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, load_in_4bit=True, device_map="auto")
extractor = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=64, device=0)

df = pd.read_pickle(DATA_PATH)
all_triples = []

for i, row in df.iterrows():
    doc_triples = []
    topic_terms = row['topic_terms']
    for sent in row['cleaned'].split('.'):
        sent = sent.strip()
        # Filtra solo oraciones que contengan términos importantes para ese tópico
        if any(t in sent for t in topic_terms):
            prompt = (
                f"Términos clave del tópico: {', '.join(topic_terms[:10])}\n"
                f"Oración: {sent}\n"
                f"Extrae todas las ternas (Sujeto, Predicado, Objeto) relevantes en formato Python list."
            )
            gen = extractor(prompt)[0]['generated_text']
            try:
                triples = eval(gen.split('\n')[-1])
            except Exception:
                triples = []
            doc_triples.extend(triples)
    all_triples.append(doc_triples)
df['triples'] = all_triples
df.to_pickle(OUT_PATH)
print(f"Guardadas ternas extraídas en {OUT_PATH}")
