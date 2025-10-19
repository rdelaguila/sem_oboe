from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import os

MODEL_PATH = "models/lora_trex"
DATA_PATH = "data/corpus_ft/preprocessed.pkl"
OUT_PATH = "data/triples_raw/triples_bbc.pkl"
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, load_in_4bit=True, device_map="auto")
extractor = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=64, device=0)

df = pd.read_pickle(DATA_PATH)
all_triples = []

for i, row in df.iterrows():
    doc_triples = []
    for sent in row['sentences']:
        if any(t in sent for t in row['tfidf_top']):
            prompt = f"Extrae todas las ternas (Sujeto, Predicado, Objeto) del texto: {sent}\n"
            gen = extractor(prompt)[0]['generated_text']
            try:
                triples = eval(gen.split('\n')[-1])
            except Exception:
                triples = []
            doc_triples.extend(triples)
    all_triples.append(doc_triples)

df['triples'] = all_triples
df.to_pickle(OUT_PATH)
