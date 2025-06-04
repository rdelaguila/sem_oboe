import pandas as pd
import nltk, re, os
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('punkt')

RAW_PATH = "data/corpus_raw/bbc.csv"
OUT_PATH = "data/corpus_ft/preprocessed.pkl"
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

df = pd.read_csv(RAW_PATH)  # columnas: doc_id, text
df['clean'] = df['text'].str.lower().apply(lambda x: re.sub(r'\s+', ' ', x))
df['sentences'] = df['clean'].apply(nltk.sent_tokenize)

vectorizer = TfidfVectorizer(max_df=0.7, min_df=5, ngram_range=(1,3))
X = vectorizer.fit_transform(df['clean'])
df['tfidf_top'] = [
    [vectorizer.get_feature_names_out()[i] for i in row.argsort()[-15:][::-1]]
    for row in X.toarray()
]

df.to_pickle(OUT_PATH)
