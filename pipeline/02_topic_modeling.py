import pandas as pd
import spacy
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
import os
import pickle

# ---------------- CONFIG ----------------
RAW_PATH = "data/corpus_raw/bbc.csv"
LDA_EVAL_DIR = "data/lda_eval/"
os.makedirs(LDA_EVAL_DIR, exist_ok=True)
ALPHAS = ['asymmetric', 'symmetric', 0.01, 0.1, 0.5]
BETAS  = ['symmetric', 0.01, 0.1, 0.5]
LIMIT = 15  # Máximo número de topics a probar
START = 2
STEP  = 1

# ---------------- PREPROCESAMIENTO (spaCy) ----------------
df = pd.read_csv(RAW_PATH)
df['cleaned'] = df['text'].astype(str).str.lower()

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
STOPWORDS = nlp.Defaults.stop_words
removal = ['ADV','PRON','CCONJ','PUNCT','PART','DET','ADP','SPACE','SYM']

print("Lemmatizando y filtrando textos...")
texts = df['cleaned'].tolist()
docs = list(nlp.pipe(texts, batch_size=32, n_process=2))
tokens = [
    [token.lemma_.lower() for token in doc if token.pos_ not in removal
        and not token.is_stop
        and token.lemma_.lower() not in STOPWORDS
        and token.is_alpha]
    for doc in docs
]
df['tokens'] = tokens

# ---------------- CORPUS Y DICCIONARIO ----------------
print("Creando diccionario y corpus Gensim...")
id2word = corpora.Dictionary(tokens)
id2word.filter_extremes(no_below=5, no_above=0.7)
corpus = [id2word.doc2bow(text) for text in tokens]

# ---------------- SELECCIÓN AUTOMÁTICA DE TOPICS ----------------
def compute_coherence_values(dictionary, corpus, texts, limit, start, step, alpha, beta):
    coherence_values = []
    model_list = []
    model_parameters = []
    for num_topics in range(start, limit+1, step):
        for a in alpha:
            for b in beta:
                model = LdaModel(corpus=corpus,
                                 id2word=dictionary,
                                 num_topics=num_topics,
                                 random_state=0,
                                 chunksize=200,
                                 alpha=a,
                                 eta=b,
                                 per_word_topics=True)
                model_list.append(model)
                coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
                coherence_values.append(coherencemodel.get_coherence())
                model_parameters.append({'num_topics':num_topics, 'alpha':a, 'eta':b})
    return model_list, coherence_values, model_parameters

print("Buscando número óptimo de topics y mejores hiperparámetros...")
model_list, coherence_values, model_parameters = compute_coherence_values(
    dictionary=id2word,
    corpus=corpus,
    texts=tokens,
    start=START,
    limit=LIMIT,
    step=STEP,
    alpha=ALPHAS,
    beta=BETAS
)

# ---------------- GUARDAR RESULTADOS DE EVALUACIÓN ----------------
eval_results = pd.DataFrame(model_parameters)
eval_results['coherence'] = coherence_values
eval_results.to_csv(os.path.join(LDA_EVAL_DIR, "lda_coherence_grid.csv"), index=False)

# Mejor modelo
best_idx = eval_results['coherence'].idxmax()
best_model = model_list[best_idx]
best_params = model_parameters[best_idx]
best_num_topics = best_params['num_topics']

print(f"Best model: topics={best_num_topics}, alpha={best_params['alpha']}, beta={best_params['eta']}, coherence={coherence_values[best_idx]:.4f}")

# Guarda el modelo, diccionario, corpus y tokens limpios
best_model.save(os.path.join(LDA_EVAL_DIR, "lda_model"))
id2word.save(os.path.join(LDA_EVAL_DIR, "dictionary.dict"))
with open(os.path.join(LDA_EVAL_DIR, "corpus.pkl"), "wb") as f:
    pickle.dump(corpus, f)
with open(os.path.join(LDA_EVAL_DIR, "tokens.pkl"), "wb") as f:
    pickle.dump(tokens, f)

# Etiqueta cada doc con topic principal y distribución
df['topic_dist'] = [best_model.get_document_topics(id2word.doc2bow(text)) for text in tokens]
df['topic'] = [max(dist, key=lambda x: x[1])[0] for dist in df['topic_dist']]
df.to_pickle(os.path.join(LDA_EVAL_DIR, "df_topic.pkl"))

# Términos más relevantes por tópico
top_terms_by_topic = {}
for topic in range(best_num_topics):
    top_terms_by_topic[topic] = [w for w, _ in best_model.show_topic(topic, topn=30)]
with open(os.path.join(LDA_EVAL_DIR, "top_terms_by_topic.pkl"), "wb") as f:
    pickle.dump(top_terms_by_topic, f)

print(f"Tópicos y evaluación guardados en: {LDA_EVAL_DIR}")
