from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import pandas as pd, pickle
import os

MODEL_PATH = "models/lora_trex"
TRIPLES_PATH = "data/triples_raw/triples_bbc.pkl"
LABELS_PATH = "data/clusters/labels_bbc.pkl"
OUT_PATH = "data/explicaciones/explicaciones_bbc.pkl"
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, load_in_4bit=True, device_map="auto")
explainer = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=128, device=0)

df = pd.read_pickle(TRIPLES_PATH)
labels = pickle.load(open(LABELS_PATH, "rb"))

explicaciones = []
for idx, row in df.iterrows():
    for t in row['triples']:
        cluster = labels[idx]
        keywords = row['tfidf_top']
        prompt = (f"La terna {t} pertenece al cluster {cluster} con palabras clave {keywords}. "
                  "Explica brevemente por qué esta terna está agrupada así, en lenguaje natural, para un humano.")
        expl = explainer(prompt)[0]['generated_text']
        explicaciones.append({
            "doc_id": row['doc_id'],
            "triple": t,
            "cluster": cluster,
            "explicacion": expl
        })

pd.DataFrame(explicaciones).to_pickle(OUT_PATH)
