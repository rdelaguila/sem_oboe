# File: topic_explanation_algorithm.py
"""
Implementación detallada del algoritmo original sin CLI: procesamiento de tripletas,
expansión de vocabulario y clustering jerárquico usando spaCy para similitud.
Se añade generación de explicaciones con Qwen-2.5 y evaluación con mT5.
Configuración en la sección OWN CONFIG.
"""
import os
import re
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import spacy
import torch
from semantic_oboe.utils.types import Tripleta
from operator import itemgetter
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, T5Tokenizer, T5ForConditionalGeneration

# ======== OWN CONFIG ========
TRIPLES_PATH      = 'data/triples.csv'       # CSV con columnas [subject, relation, object, topic]
TOPIC_ID          = 0                        # ID del tópico a procesar
VISITAR_OBJETO    = True                     # incluir objetos en el filtrado
TERMINOS_A_INCLUIR = None                    # set de términos relevantes (None para todos)
DICTDBP_PATH      = 'data/dbpedia_dict.json' # JSON precomputado de mapeo dbpedia
DICTNER_PATH      = 'data/ner_dict.json'     # JSON precomputado de entidades NER
N_SINONIMOS       = 1                        # número de tipos más similares a retener
OUTPUT_DIR        = 'output'
SPACY_MODEL       = 'es_core_news_md'        # modelo spaCy para similitud
# Qwen 2.5 generación
gen_model_name    = 'Qwen/Qwen-2.5B'
# mT5 evaluación
eval_model_name   = 'google/mt5-small'
# ============================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Carga recursos
df_tr = pd.read_csv(TRIPLES_PATH)
with open(DICTDBP_PATH, 'r', encoding='utf-8') as f:
    dictdbp = json.load(f)
with open(DICTNER_PATH, 'r', encoding='utf-8') as f:
    dictner = json.load(f)

nlp = spacy.load(SPACY_MODEL)

def remove_numbers(text): return re.sub(r"\d+", "", text)

def remove_dbpedia_categories(s): return s.split('/')[-1]

def return_url_element(s):
    for sep in ['#','/']:
        if sep in s:
            s = s.split(sep)[-1]
    return s

# 1. Filtrado y extracción de tripletas relevantes
listado_tripletas = []
anterior = None
for _, row in df_tr.iterrows():
    trip = Tripleta({'subject': str(row['subject']),
                     'relation': row['relation'],
                     'object': str(row['object'])})
    if anterior is None:
        anterior = trip
    misma_super = (trip.esTripletaSuper(anterior) == anterior.esTripletaSuper(trip))
    dif = trip.dondeSonDiferentes(anterior)
    if not (misma_super and (dif == ('sujeto','relacion','objeto') or dif == ('sujeto',None,'objeto'))):
        anterior = trip
    else:
        continue

    sujeto = set(trip.sujeto.split())
    objeto = set(trip.objeto.split()) if VISITAR_OBJETO else set()
    if TERMINOS_A_INCLUIR and sujeto.isdisjoint(TERMINOS_A_INCLUIR) and objeto.isdisjoint(TERMINOS_A_INCLUIR):
        continue

    visitados = set()
    for termino in sujeto.union(objeto):
        if termino in visitados or termino[0].isdigit():
            continue
        info_list = dictdbp.get(termino.lower(), [])
        if not info_list:
            continue
        info = info_list[0]
        tipos_dbp = info.get('tipos', [])
        uri = info.get('URI', '')
        # WordNet
        import nltk
        from nltk.corpus import wordnet as wn
        nltk.download('wordnet', quiet=True)
        sinonimos, lwordnet = [], []
        for syn in wn.synsets(termino):
            sinonimos.extend(syn.lemma_names())
            for h in syn.hypernyms(): lwordnet.extend(h.lemma_names())
        ner_ent = dictner.get(termino.lower(), '')
        listado_tripletas.append({
            'termino': termino,
            'sinonimos': list(set(sinonimos)),
            'resource': uri,
            'dbpedia': tipos_dbp,
            'ner': ner_ent,
            'wordnet': lwordnet
        })
        visitados.add(termino)

# 2. Expandir vocabulario con SpaCy similarity
df = pd.DataFrame(listado_tripletas)
vocab_aux, lista_tipos = [], []
for _, row in df.iterrows():
    termino = row['termino']
    tipos = []
    tipos.extend(row['dbpedia'] if isinstance(row['dbpedia'], list) else row['dbpedia'].split(','))
    tipos.extend(row['wordnet'])
    tipos_clean = []
    for t in tipos:
        el = return_url_element(remove_dbpedia_categories(remove_numbers(str(t))))
        if el and el != 'Q': tipos_clean.append(el)
    sims = [nlp(termino).similarity(nlp(t2)) for t2 in tipos_clean]
    if not sims: continue
    idx = list(np.argpartition(sims, -N_SINONIMOS)[-N_SINONIMOS:])
    sel = itemgetter(*idx)(tipos_clean)
    lista_tipos.append({'termino': termino, 'tipos': sel})
    vocab_aux.append(termino)
    if isinstance(sel, str): vocab_aux.append(sel)
    else: vocab_aux.extend(sel)
vocab = set([token.lemma_.lower() for token in nlp.pipe(vocab_aux)])
terms = list(vocab)
M = np.array([[nlp(t1).similarity(nlp(t2)) for t2 in terms] for t1 in terms])

# 3. Clustering jerárquico y selección k óptimo
plt.figure(figsize=(10,7)); plt.title("Dendrograma de términos")
shc.dendrogram(shc.linkage(M, method='ward', optimal_ordering=True), labels=terms, leaf_rotation=90)
plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR,'dendrogram.png')); plt.close()
range_k = range(2, min(len(terms)-1, 10))
sil = []
for k in range_k:
    labs = AgglomerativeClustering(n_clusters=k).fit_predict(M)
    sil.append(silhouette_score(M, labs))
best_k = range_k[np.argmax(sil)]
labs = AgglomerativeClustering(n_clusters=best_k).fit_predict(M)
clusters = {i: [] for i in set(labs)}
for t, l in zip(terms, labs): clusters[l].append(t)
with open(os.path.join(OUTPUT_DIR,'clusters.json'),'w',encoding='utf-8') as f:
    json.dump(clusters, f, ensure_ascii=False, indent=2)

# 4. Generación de explicaciones con Qwen-2.5
# Inicializar modelo
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tok_gen = AutoTokenizer.from_pretrained(gen_model_name)
mod_gen = AutoModelForCausalLM.from_pretrained(gen_model_name).to(torch_device)
gen_pipe = pipeline('text-generation', model=mod_gen, tokenizer=tok_gen, device=0 if torch_device=='cuda' else -1)
explanations = {}
for cid, terms_c in clusters.items():
    prompt = f"Eres un experto en NLP. Explica el tópico {TOPIC_ID}-{cid} con estos términos: {', '.join(terms_c)}."
    resp = gen_pipe(prompt, max_length=200, do_sample=False)
    explanations[cid] = resp[0]['generated_text']
with open(os.path.join(OUTPUT_DIR,'explanations.json'),'w',encoding='utf-8') as f:
    json.dump(explanations, f, ensure_ascii=False, indent=2)

# 5. Evaluación de explicaciones con mT5
# Inicializar modelo de evaluación
tok_eval = T5Tokenizer.from_pretrained(eval_model_name)
mod_eval = T5ForConditionalGeneration.from_pretrained(eval_model_name).to(torch_device)
eval_pipe = pipeline('text2text-generation', model=mod_eval, tokenizer=tok_eval, device=0 if torch_device=='cuda' else -1)
evaluations = {}
for cid, exp in explanations.items():
    prompt = (f"Evalúa la explicación para el tópico {TOPIC_ID}-{cid}. "
              f"Términos: {', '.join(clusters[cid])}. "
              f"Explicación: {exp}. "
              f"Devuelve JSON con puntuaciones 1-5 para coherencia, relevancia y cobertura.")
    resp = eval_pipe(prompt, max_length=100)
    evaluations[cid] = resp[0]['generated_text']
with open(os.path.join(OUTPUT_DIR,'evaluations.json'),'w',encoding='utf-8') as f:
    json.dump(evaluations, f, ensure_ascii=False, indent=2)

# 6. Texto resumen
lines = [
    f"Para el tópico {TOPIC_ID} se generaron {best_k} temas tras clustering y se evaluaron las explicaciones con mT5."
]
for cid, terms_c in clusters.items():
    lines.append(f"Tema {cid}: {', '.join(terms_c)}")
with open(os.path.join(OUTPUT_DIR,'summary.txt'),'w',encoding='utf-8') as f:
    f.write("\n".join(lines))

print(f"Pipeline completado. Salida en {OUTPUT_DIR}")
