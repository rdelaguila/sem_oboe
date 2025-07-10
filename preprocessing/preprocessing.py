
import pandas as pd
import spacy
import joblib
import neuralcoref
from joblib import Parallel, delayed
import re

# Load SpaCy and add neuralcoref
nlp = spacy.load("en_core_web_sm")
neuralcoref.add_to_pipe(nlp)

def resolve_coref(doc):
    return doc._.coref_resolved

def process_chunk_corefs(docs):
    coref_texts = []
    for doc in nlp.pipe(docs, batch_size=20):
        coref_texts.append(resolve_coref(doc))
    return coref_texts

def token_filter(token):
    return not (token.is_stop or token.is_punct or token.like_num or token.like_url)

def clean(text, lemma=False):
    doc = nlp(text)
    if lemma:
        tokens = " ".join(token.lemma_ for token in doc if token_filter(token))
    else:
        tokens = " ".join(token.text for token in doc if token_filter(token))
    return tokens

def chunker(iterable, total_length, chunksize):
    return (iterable[pos: pos + chunksize] for pos in range(0, total_length, chunksize))

def flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]

def preprocess_parallel_corefs(texts, chunksize=100):
    executor = Parallel(n_jobs=7, backend='multiprocessing', prefer="processes")
    do = delayed(process_chunk_corefs)
    tasks = (do(chunk) for chunk in chunker(texts, len(texts), chunksize=chunksize))
    result = executor(tasks)
    return flatten(result)

def to_nlp(text):
    return nlp(text)
