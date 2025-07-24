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
from sklearn.metrics import silhouette_score, silhouette_samples
import spacy
import torch
from utils.types import StringCaseInsensitiveSet, CaseInsensitiveDict, CaseInsensitiveSet
from utils.triplet_manager_lib import Tripleta
from operator import itemgetter
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, T5Tokenizer, T5ForConditionalGeneration
import joblib

# ======== OWN CONFIG ========
TRIPLES_PATH      = "data/triples_ft/processed/dataset_final_triplet_bbc_pykeen"
# CSV con columnas [subject, relation, object, topic]
TOPIC_ID          = 3                       # ID del tópico a procesar
VISITAR_OBJETO    = True                     # incluir objetos en el filtrado
TERMINOS_A_INCLUIR  = set(['dvd','google','electronic','tv','sony','screen','nintendo','player','mobile','phone','software','video','network','apple','program','linux'])                    # set de términos relevantes (None para todos)
DBPEDIA_PATH = 'data/corpus_ft/bbc/diccionario_topic_entidades_dbpedia'
DICTNER_PATH      = 'data/corpus_ft/bbc/diccionario_ner'     # JSON precomputado de entidades NER
N_SINONIMOS       = 1                        # número de tipos más similares a retener
OUTPUT_DIR        = 'output'
SPACY_MODEL       = 'en_core_web_lg'        # modelo spaCy para similitud
# Qwen 2.5 generación
gen_model_name    = 'Qwen/Qwen2-7B-Instruct'
# mT5 evaluación
eval_model_name   = 'google/mt5-small'
nlp = spacy.load(SPACY_MODEL)
import nltk
from nltk.corpus import wordnet as wn

# ============================ REVISAR
topics_entsdbpedia = joblib.load(DBPEDIA_PATH)
diccionario_dbpedia = topics_entsdbpedia.get(TOPIC_ID, {})

dictdbp = diccionario_dbpedia
dictner = joblib.load(DICTNER_PATH)
os.makedirs(OUTPUT_DIR, exist_ok=True)
# Carga recursos
df_tr = joblib.load(TRIPLES_PATH)
print (df_tr.shape)
nltk.download('wordnet', quiet=True)
def remove_numbers(text): return re.sub(r"\d+", "", text)

def remove_dbpedia_categories(s): return s.split('/')[-1]

def return_url_element(s):
    for sep in ['#','/']:
        if sep in s:
            s = s.split(sep)[-1]
    return s

# 1. Filtrado y extracción de tripletas relevantes
listado_tripletas = []
palabrasdbpedia = set(k.lower() for k in dictdbp.keys()) ## aqui
print("dictdbp:", dictdbp)
print("len(dictdbp):", len(dictdbp))
print("Claves ejemplo:", list(dictdbp.keys())[:10])
anterior = None
for i, row in df_tr.iterrows():
    tripleta = Tripleta({'subject': str(row['subject']),
                     'relation': row['relation'],
                     'object': str(row['object'])})

    sujeto = set(tripleta.sujeto.split())
    objeto = set(tripleta.objeto.split()) if VISITAR_OBJETO else set()

    # Si es la primera iteración
    if anterior is None:
        anterior = tripleta

    # Comparación entre tripletas
    misma_super = (tripleta.esTripletaSuper(anterior) == anterior.esTripletaSuper(tripleta))
    dif = tripleta.dondeSonDiferentes(anterior)

    if (misma_super and (dif == ('sujeto', 'relacion', 'objeto') or dif == ('sujeto', None, 'objeto'))):
        anterior = tripleta
    else:
        continue

    # Filtro por términos a incluir
    if (TERMINOS_A_INCLUIR is None
            or not TERMINOS_A_INCLUIR.isdisjoint(sujeto)
            or (VISITAR_OBJETO and not TERMINOS_A_INCLUIR.isdisjoint(objeto))):

        visitados = set()

        # Crea set de términos existentes en dictdbp (en minúsculas)

        encontradas = sujeto.intersection(palabrasdbpedia)
        no_encontradas = sujeto.difference(palabrasdbpedia)

        if VISITAR_OBJETO:
            encontradas.update(objeto.intersection(palabrasdbpedia))
            no_encontradas.update(objeto.difference(palabrasdbpedia))

        final = encontradas.union(no_encontradas)

        for termino in encontradas:
            termino_lower = termino.lower()

            if termino in visitados:
                continue

            if termino[0].isdigit():
                no_encontradas.add(termino)
                continue

            info_list = dictdbp.get(termino_lower, [])
            if not info_list:
                no_encontradas.add(termino)
                continue

            info_termino = info_list[0]
            uri_db = info_termino.get('URI', '')
            tipos_db = info_termino.get('tipos', [])

            sinonimos = []
            lwordnet = []

            # WordNet synonyms + hypernyms
            for syn in wn.synsets(termino):
                sinonimos.extend(syn.lemma_names())
                for h in syn.hypernyms():
                    lwordnet.extend(h.lemma_names())

            # NER
            sujeto_en_ner = dictner.get(termino_lower, '')
            ner = []
            if sujeto_en_ner:
                ner.append(sujeto_en_ner)

            diccionario_termino = {
                'termino': termino,
                'sinonimos': list(set(sinonimos)),
                'resource': uri_db,
                'dbpedia': tipos_db,
                'ner': ner,
                'wordnet': lwordnet
            }

            listado_tripletas.append(diccionario_termino)
            visitados.add(termino)
df = pd.DataFrame(listado)
vocab_aux=[]
for _,row in df.iterrows():
    term=row.termino; tips=row.dbpedia+row.wordnet
    clean=[clean_label(t) for t in tips if clean_label(t)!='Q']
    sims=[nlp(term).similarity(nlp(c)) for c in clean]
    if not sims: continue
    idx=list(np.argpartition(sims,-N_SINONIMOS)[-N_SINONIMOS:])
    sel=itemgetter(*idx)(clean)
    vocab_aux.append(term)
    vocab_aux += sel if isinstance(sel,(list,tuple)) else [sel]
vocab=set(t.lemma_.lower() for t in nlp.pipe(vocab_aux))
terms=list(vocab)
M=np.array([[nlp(t1).similarity(nlp(t2)) for t2 in terms] for t1 in terms])

# 3. Clustering jerárquico + silhouette
plt.figure(figsize=(10,7)); plt.title('Dendrograma');
shc.dendrogram(shc.linkage(M,method='ward',optimal_ordering=True),labels=terms,leaf_rotation=90)
plt.savefig(os.path.join(OUTPUT_DIR,'dendro.png')); plt.close()
range_k=range(2,min(len(terms)-1,10))
sil=[silhouette_score(M,AgglomerativeClustering(n_clusters=k).fit_predict(M)) for k in range_k]
bk=range_k[np.argmax(sil)]; labels=AgglomerativeClustering(n_clusters=bk).fit_predict(M)
clusters={i:[t for t,l in zip(terms,labels) if l==i] for i in set(labels)}
global_sil=silhouette_score(M,labels)
with open(os.path.join(OUTPUT_DIR,'clusters.json'),'w',encoding='utf-8') as f:
    json.dump({'best_k':int(bk),'global_sil':global_sil,'clusters':clusters},f,indent=2)

# 4. Generación con Qwen-2.5
device=0 if torch.cuda.is_available() else -1
tok_g=AutoTokenizer.from_pretrained(GEN_MODEL)
mod_g=AutoModelForCausalLM.from_pretrained(GEN_MODEL).to('cuda' if torch.cuda.is_available() else 'cpu')
gen= pipeline('text-generation',model=mod_g,tokenizer=tok_g,device=device)
exps={}
for cid,terms_c in clusters.items():
    prompt=(f"Explica en español y conciso el cluster {TOPIC_ID}-{cid}. "
            f"Términos: {', '.join(terms_c)}. Silhouette: "
            f"cluster={silhouette_score(M,labels):.3f}, global={global_sil:.3f}.")
    out=gen(prompt, max_new_tokens=150, do_sample=False, return_full_text=False)
    exps[cid]=out[0]['generated_text']
with open(os.path.join(OUTPUT_DIR,'explanations.json'),'w',encoding='utf-8') as f:
    json.dump(exps,f,ensure_ascii=False,indent=2)

# 5. Evaluación con mT5
tok_e=T5Tokenizer.from_pretrained(EVAL_MODEL)
mod_e=T5ForConditionalGeneration.from_pretrained(EVAL_MODEL).to('cuda' if torch.cuda.is_available() else 'cpu')
evalp=pipeline('text2text-generation',model=mod_e,tokenizer=tok_e,device=device)
evals={}
for cid,exp in exps.items():
    terms_c=clusters[cid]
    prompt=(f"Evalúa la explicación del cluster {TOPIC_ID}-{cid}. "
            f"Términos: {', '.join(terms_c)}. Explicación: {exp}."
            " Devuelve JSON con puntuaciones del 1 al 5 para coherencia, relevancia y cobertura.")
    out=evalp(prompt, max_new_tokens=80, do_sample=False, return_full_text=False)
    try:
        evals[cid]=json.loads(out[0]['generated_text'])
    except:
        evals[cid]=out[0]['generated_text']
with open(os.path.join(OUTPUT_DIR,'evaluations.json'),'w',encoding='utf-8') as f:
    json.dump(evals,f,ensure_ascii=False,indent=2)

# 6. Resumen final
with open(os.path.join(OUTPUT_DIR,'summary.txt'),'w',encoding='utf-8') as f:
    f.write(f"Tópico {TOPIC_ID}: {bk} clusters; silhouette global {global_sil:.3f}\n")
    for cid,terms_c in clusters.items():
        f.write(f"Cluster {cid} (sil={evals[cid].get('silhouette',0) if isinstance(evals[cid],dict) else 'N/A'}): ")
        f.write(', '.join(terms_c)+"\n")
print('Pipeline completo. Revisar output.')
