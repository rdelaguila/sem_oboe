import pandas as pd
import pickle
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import os

TRIPLES_PATH = "data/triples_raw/triples_bbc_semantic.pkl"
CLUSTERS_PATH = "data/clusters/labels_bbc.pkl"
OUT_PATH = "data/explicaciones/explicaciones_inferidas.pkl"
MODEL_PATH = "models/lora_trex"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, load_in_4bit=True, device_map="auto")
explainer = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=128, device=0)

df = pd.read_pickle(TRIPLES_PATH)
labels = pickle.load(open(CLUSTERS_PATH, "rb"))

inferidas, concept_cluster, idx = [], {}, 0
for row in df['triples_semantics']:
    for t in row:
        concept_cluster[t['s']] = labels[idx]
        idx += 1
        concept_cluster[t['o']] = labels[idx]
        idx += 1

# Infere nuevas ternas entre clusters distintos pero sinónimos o URIs compartidas
def find_synonym_clusters(df):
    entities, idx = [], 0
    for row in df['triples_semantics']:
        for t in row:
            entities.append({'entidad': t['s'], 'uri': t['s_uri'], 'syns': set(t['s_syns']), 'cluster': labels[idx]})
            idx += 1
            entities.append({'entidad': t['o'], 'uri': t['o_uri'], 'syns': set(t['o_syns']), 'cluster': labels[idx]})
            idx += 1
    ternas_inf = []
    for i, e1 in enumerate(entities):
        for j, e2 in enumerate(entities):
            if i >= j: continue
            if e1['cluster'] == e2['cluster']: continue
            if (e1['uri'] and e1['uri'] == e2['uri']) or (e1['syns'] & e2['syns']):
                ternas_inf.append((e1['entidad'], 'relacionada_con', e2['entidad'], e1['cluster'], e2['cluster']))
    return ternas_inf

ternas_inferidas = find_synonym_clusters(df)
explicaciones = []
for s, r, o, c1, c2 in ternas_inferidas:
    prompt = (f"La terna inferida ({s}, {r}, {o}) conecta el cluster {c1} y el cluster {c2}, "
              f"porque '{s}' y '{o}' comparten URI o sinónimos. "
              "Explica brevemente, en español, por qué tiene sentido inferir esta relación y qué aporta a la comprensión del grafo.")
    expl = explainer(prompt)[0]['generated_text']
    explicaciones.append({
        "triple_inferida": (s, r, o),
        "clusters": (c1, c2),
        "explicacion": expl
    })
pd.DataFrame(explicaciones).to_pickle(OUT_PATH)
