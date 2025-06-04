import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

MODEL = SentenceTransformer("all-MiniLM-L6-v2")
TRIPLES_PATH = "data/triples_raw/triples_bbc_semantic.pkl"
OUT_PATH = "data/clusters/labels_semantic.pkl"

df = pd.read_pickle(TRIPLES_PATH)
all_concepts = []
for row in df['triples_semantics']:
    for t in row:
        # Junta las definiciones y sinónimos para expandir la representación del concepto
        s_expanded = t['s'] + " " + (t['s_def'] or '') + " " + " ".join(t['s_syns'])
        o_expanded = t['o'] + " " + (t['o_def'] or '') + " " + " ".join(t['o_syns'])
        all_concepts.append(s_expanded)
        all_concepts.append(o_expanded)
embeddings = MODEL.encode(all_concepts)
Z = linkage(embeddings, method='ward')
labels = fcluster(Z, 4, criterion='maxclust')  # Ejemplo: 4 clusters, ajusta a lo óptimo
with open(OUT_PATH, "wb") as f:
    import pickle
    pickle.dump(labels, f)
