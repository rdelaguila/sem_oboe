import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import joblib

TRIPLES_PATH = "data/triples_raw/triples_bbc.pkl"
EMB_PATH = "data/triples_emb/entity_embeddings.npy"
REL_EMB_PATH = "data/triples_emb/relation_embeddings.npy"
TFIDF_PATH = "data/corpus_ft/preprocessed.pkl"
OUT_PATH = "data/triples_raw/triples_clasificadas.pkl"

df = pd.read_pickle(TRIPLES_PATH)
entity_embs = np.load(EMB_PATH)
rel_embs = np.load(REL_EMB_PATH)
df_tfidf = pd.read_pickle(TFIDF_PATH)

# Asumimos que los índices se corresponden (sujeto, predicado, objeto)
features = []
labels = []

# EJEMPLO: añade tu propio etiquetado manual para entrenamiento
for idx, row in df.iterrows():
    for t in row['triples']:
        s, r, o = t
        # Busca el embedding correspondiente para cada término (si existe)
        # Esto requiere que tengas mapeo de entidades a IDs de embeddings
        # Por simplicidad, aquí solo mostramos la estructura:
        # e1_vec, rel_vec, e2_vec = buscar_emb(s), buscar_emb(r), buscar_emb(o)
        # tfidf_sum = suma de tfidf de las palabras s, r, o del documento
        features.append(np.concatenate([e1_vec, rel_vec, e2_vec, [tfidf_sum]]))
        labels.append(row.get("etiqueta", 1))  # 1 = válida, 0 = ruido (manual o automático)

X = np.array(features)
y = np.array(labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)
clf = RandomForestClassifier(n_estimators=200)
clf.fit(X_train, y_train)
print("Score:", clf.score(X_test, y_test))
joblib.dump(clf, "data/triples_raw/triple_clf.joblib")
# Clasifica todo el corpus y añade columna de plausibilidad
probs = clf.predict_proba(X)[:,1]
df['plausibilidad'] = probs
df.to_pickle(OUT_PATH)
