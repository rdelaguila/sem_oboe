import pandas as pd
import pickle
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import numpy as np
import os

# Paths
TRIPLES_PATH = "data/triples_raw/triples_bbc_semantic.pkl"
LABELS_PATH = "data/clusters/labels_bbc.pkl"
CLUSTER_EVAL_PATH = "data/clusters/clustering_eval.pkl"  # Debes generar este con Silhouette por cluster
OUT_PATH = "data/explicaciones/explicaciones_bbc.pkl"
MODEL_PATH = "models/lora_trex"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, load_in_4bit=True, device_map="auto")
explainer = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=180, device=0)

df = pd.read_pickle(TRIPLES_PATH)
labels = pickle.load(open(LABELS_PATH, "rb"))
# Debes tener una estructura como {cluster: silhouette_score}
if os.path.exists(CLUSTER_EVAL_PATH):
    with open(CLUSTER_EVAL_PATH, "rb") as f:
        cluster_eval = pickle.load(f)
else:
    cluster_eval = {}  # Si no existe, solo incluye Silhouette global

# Carga palabras clave/top terms por cluster
LDA_EVAL_DIR = "data/lda_eval/"
with open(os.path.join(LDA_EVAL_DIR, "top_terms_by_topic.pkl"), "rb") as f:
    top_terms_by_topic = pickle.load(f)

# Carga Silhouette global (lo puedes obtener de tu script de clustering)
global_silhouette = np.mean(list(cluster_eval.values())) if cluster_eval else None

# ---- Generación de explicaciones ----
explicaciones = []
doc_idx = 0
for idx, row in df.iterrows():
    triples = row.get('triples', [])
    triples_semantics = row.get('triples_semantics', [])
    if not triples_semantics: continue
    for t_sem in triples_semantics:
        # Info de la terna
        s, r, o = t_sem["s"], t_sem["r"], t_sem["o"]
        # Cluster asignado (asume que hay tantos labels como entidades/triples)
        cluster_id = labels[doc_idx % len(labels)]
        cluster_terms = top_terms_by_topic.get(cluster_id, [])
        silhouette = cluster_eval.get(cluster_id, None)
        # Construye el prompt con TODA la info relevante
        prompt = (
            f"Tema/cluster asignado a la terna: {cluster_id}\n"
            f"Palabras clave del cluster: {', '.join(cluster_terms)}\n"
            f"Medida de calidad del cluster (Silhouette): {silhouette if silhouette is not None else 'N/A'} "
            f"(Silhouette global: {global_silhouette if global_silhouette else 'N/A'})\n"
            f"Terna: ({s}, {r}, {o})\n"
            f"Explica en español, de forma clara y concisa, por qué esta terna pertenece a este cluster, "
            f"cómo se justifica según las palabras clave, y qué nos dice la medida de calidad sobre la coherencia del agrupamiento. "
            f"Si la Silhouette es baja, explica posibles problemas; si es alta, destaca la cohesión semántica."
        )
        expl = explainer(prompt)[0]['generated_text']
        explicaciones.append({
            "terna": (s, r, o),
            "cluster": cluster_id,
            "palabras_cluster": cluster_terms,
            "silhouette": silhouette,
            "explicacion": expl
        })
        doc_idx += 1

pd.DataFrame(explicaciones).to_pickle(OUT_PATH)
print(f"Guardadas {len(explicaciones)} explicaciones en {OUT_PATH}")
