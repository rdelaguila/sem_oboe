import pickle
import pandas as pd
from collections import defaultdict

CLUSTERS_PATH = "data/clusters/labels_semantic.pkl"
TRIPLES_PATH = "data/triples_raw/triples_bbc_semantic.pkl"
OUT_PATH = "data/clusters/clusters_expanded.pkl"

labels = pickle.load(open(CLUSTERS_PATH, "rb"))
df = pd.read_pickle(TRIPLES_PATH)
cluster2entities = defaultdict(set)
entity2cluster = {}

# Asocia cada entidad expandida a un cluster
idx = 0
for row in df['triples_semantics']:
    for t in row:
        cluster2entities[labels[idx]].add(t['s'])
        entity2cluster[t['s']] = labels[idx]
        idx += 1
        cluster2entities[labels[idx]].add(t['o'])
        entity2cluster[t['o']] = labels[idx]
        idx += 1

# Expande clusters: si dos entidades son sinónimas o comparten URI, fusiona clusters
def are_synonyms(syns1, syns2):
    return bool(set(syns1) & set(syns2))

merged = defaultdict(set)
for c1, ents1 in cluster2entities.items():
    for c2, ents2 in cluster2entities.items():
        if c1 >= c2:
            continue
        # Si hay solapamiento de sinónimos, fusiona clusters
        for e1 in ents1:
            for e2 in ents2:
                # Requiere acceso a sinónimos: aquí deberías recuperar los sinónimos de cada entidad
                pass  # Lógica de fusión aquí
# Almacena los clusters expandidos
with open(OUT_PATH, "wb") as f:
    pickle.dump(cluster2entities, f)
