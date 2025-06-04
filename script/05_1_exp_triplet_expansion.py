import pandas as pd
import requests
import nltk
from nltk.corpus import wordnet as wn
nltk.download('wordnet')
nltk.download('omw-1.4')

TRIPLES_PATH = "data/triples_raw/triples_bbc.pkl"
OUT_PATH = "data/triples_raw/triples_bbc_semantic.pkl"

def dbpedia_spotlight(entity):
    url = "https://api.dbpedia-spotlight.org/en/annotate"
    headers = {'Accept': 'application/json'}
    params = {"text": entity, "confidence": 0.5}
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=5)
        if resp.status_code == 200:
            resources = resp.json().get("Resources", [])
            if resources:
                return resources[0]['@URI']
    except Exception:
        return None
    return None

def wordnet_synsets(entity):
    synsets = wn.synsets(entity)
    if synsets:
        return synsets[0].definition(), [l.name() for l in synsets[0].lemmas()]
    return None, []

df = pd.read_pickle(TRIPLES_PATH)
expansions = []
for idx, row in df.iterrows():
    expanded = []
    for t in row['triples']:
        s, r, o = t
        s_uri = dbpedia_spotlight(s)
        o_uri = dbpedia_spotlight(o)
        s_def, s_syns = wordnet_synsets(s)
        o_def, o_syns = wordnet_synsets(o)
        expanded.append({
            "s": s, "r": r, "o": o,
            "s_uri": s_uri, "o_uri": o_uri,
            "s_def": s_def, "s_syns": s_syns,
            "o_def": o_def, "o_syns": o_syns,
        })
    expansions.append(expanded)
df['triples_semantics'] = expansions
df.to_pickle(OUT_PATH)
