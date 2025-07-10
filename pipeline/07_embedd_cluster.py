import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score
import pickle
import os

EMB_PATH = "data/triples_emb/entity_embeddings.npy"
LABELS_PATH = "data/clusters/labels_bbc.pkl"
os.makedirs(os.path.dirname(LABELS_PATH), exist_ok=True)

X = np.load(EMB_PATH)
Z = linkage(X, method='ward')
# Selección óptima de clusters
best_k, best_sil = 2, -1
for k in range(2, 10):
    labels = fcluster(Z, k, criterion='maxclust')
    sil = silhouette_score(X, labels)
    print(f"k={k} sil={sil:.3f}")
    if sil > best_sil:
        best_sil, best_k = sil, k
labels = fcluster(Z, best_k, criterion='maxclust')
pickle.dump(labels, open(LABELS_PATH, "wb"))
