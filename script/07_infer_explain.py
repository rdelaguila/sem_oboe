import pandas as pd
import pickle
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# Rutas y carga de recursos
TRIPLES_PATH = "data/triples_raw/triples_bbc_semantic.pkl"
CLUSTERS_PATH = "data/clusters/labels_semantic.pkl"
OUT_PATH = "data/explicaciones/explicaciones_inferidas.pkl"
MODEL_PATH = "models/lora_trex"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, load_in_4bit=True, device_map="auto")
explainer = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=128, device=0)

df = pd.read_pickle(TRIPLES_PATH)
labels = pickle.load(open(CLUSTERS_PATH, "rb"))

# 1. Inferencia: generación de nuevas ternas por clusters (por sinonimia/URI compartida)
inferidas = []
concept_cluster = dict()
idx = 0

# Asignación rápida: cada concepto a un cluster
for row in df['triples_semantics']:
    for t in row:
        concept_cluster[t['s']] = labels[idx]
        idx += 1
        concept_cluster[t['o']] = labels[idx]
        idx += 1

# Búsqueda: si dos sujetos diferentes, en clusters distintos, comparten algún sinónimo o URI, infiere terna "relacionada_con"
def find_synonym_clusters(df):
    # Mapeo entidad -> (definiciones, sinónimos, URI, cluster)
    entities = []
    idx = 0
    for row in df['triples_semantics']:
        for t in row:
            entities.append({
                'entidad': t['s'],
                'uri': t['s_uri'],
                'syns': set(t['s_syns']),
                'cluster': labels[idx]
            })
            idx += 1
            entities.append({
                'entidad': t['o'],
                'uri': t['o_uri'],
                'syns': set(t['o_syns']),
                'cluster': labels[idx]
            })
            idx += 1
    # Infere nuevas ternas entre clusters distintos pero conceptos sinónimos o misma URI
    ternas_inf = []
    for i, e1 in enumerate(entities):
        for j, e2 in enumerate(entities):
            if i >= j: continue
            if e1['cluster'] == e2['cluster']: continue
            if (e1['uri'] and e1['uri'] == e2['uri']) or (e1['syns'] & e2['syns']):
                # Relación inferida
                ternas_inf.append((e1['entidad'], 'relacionada_con', e2['entidad'], e1['cluster'], e2['cluster']))
    return ternas_inf

ternas_inferidas = find_synonym_clusters(df)

# 2. Generación de explicación automática para cada terna inferida
explicaciones = []
for s, r, o, c1, c2 in ternas_inferidas:
    prompt = (
        f"La terna inferida ({s}, {r}, {o}) conecta el cluster {c1} y el cluster {c2}, "
        f"porque '{s}' y '{o}' comparten URI o sinónimos. "
        "Explica brevemente, en español, por qué tiene sentido inferir esta relación y qué aporta a la comprensión del grafo."
    )
    expl = explainer(prompt)[0]['generated_text']
    explicaciones.append({
        "triple_inferida": (s, r, o),
        "clusters": (c1, c2),
        "explicacion": expl
    })

# Guarda resultados
pd.DataFrame(explicaciones).to_pickle(OUT_PATH)
print(f"Generadas {len(explicaciones)} ternas inferidas con explicación.")

# Opcional: muestra algunos ejemplos
for row in explicaciones[:3]:
    print("Terna inferida:", row["triple_inferida"])
    print("Explicación:", row["explicacion"])
    print("-----")
