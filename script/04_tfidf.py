import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import pickle

LDA_EVAL_DIR = "data/lda_eval/"
OUT_PATH = "data/corpus_ft/preprocessed_tfidf.pkl"

# Carga el DataFrame con tópicos
df = pd.read_pickle(os.path.join(LDA_EVAL_DIR, "df_topic.pkl"))
with open(os.path.join(LDA_EVAL_DIR, "top_terms_by_topic.pkl"), "rb") as f:
    top_terms_by_topic = pickle.load(f)

# Reconstruye los textos filtrados (token a string)
df['filtered_text'] = df['tokens'].apply(lambda x: ' '.join(x))

# TF-IDF global
vectorizer = TfidfVectorizer(max_df=0.7, min_df=5, ngram_range=(1,3))
X = vectorizer.fit_transform(df['filtered_text'])
df['tfidf_top'] = [
    [vectorizer.get_feature_names_out()[i] for i in row.argsort()[-15:][::-1]]
    for row in X.toarray()
]

# Añade términos top por tópico
df['topic_terms'] = df['topic'].map(top_terms_by_topic)
# Puedes guardar la matriz tfidf si la usas downstream
import numpy as np
np.save(os.path.join(LDA_EVAL_DIR, "tfidf_matrix.npy"), X.toarray())

df.to_pickle(OUT_PATH)
print("TF-IDF y términos por tópico guardados en", OUT_PATH)
