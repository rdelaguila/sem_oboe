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

# Si quieres depuración:
print(listado_tripletas)

print (listado_tripletas)
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
    tipos_mas_similares = itemgetter (*list(np.argpartition(sims, -N_SINONIMOS)[-N_SINONIMOS:]))(tipos_clean)
    puntuaciones_tipos_similares = itemgetter(*idx)(sims)

    #vocabulario_terminos.append(termino)
    #if type(tipos_mas_similares) == str:
    #    vocabulario_terminos.append(tipos_mas_similares)
    #else:
    #    vocabulario_terminos.extend(tipos_mas_similares)
    #visitado.append(row.termino)
    lista_tipos.append({'termino': termino, 'tipos': sel,'similitudes': puntuaciones_tipos_similares})
    vocab_aux.append(termino)
    if isinstance(sel, str): vocab_aux.append(sel)
    else: vocab_aux.extend(sel)
vocab = set()

for doc in nlp.pipe(vocab_aux):
    lemmatized = " ".join([token.lemma_.lower() for token in doc])
    vocab.add(lemmatized)

terms = list(vocab)
M = np.array([[nlp(t1).similarity(nlp(t2)) for t2 in terms] for t1 in terms])

# 3. Clustering jerárquico y selección k óptimo
plt.figure(figsize=(10,7)); plt.title("Dendrograma de términos")
shc.dendrogram(shc.linkage(M, method='ward', optimal_ordering=True), labels=terms, leaf_rotation=90)
plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR,'dendrogram.png')); plt.close()
range_k = range(2, min(len(terms)-1, 10))
sil = []
for k in range_k:
    #labs = AgglomerativeClustering(n_clusters=k).fit_predict(M)
    #sil.append(silhouette_score(M, labs))
#0best_k = range_k[np.argmax(sil)]
#labs = AgglomerativeClustering(n_clusters=best_k).fit_predict(M)
#clusters = {i: [] for i in set(labs)}
#for t, l in zip(terms, labs): clusters[l].append(t)
#clusters_clean = {str(k): v for k, v in clusters.items()}

# 1. Calculamos la silueta media para cada k

    labels_k = AgglomerativeClustering(n_clusters=k).fit_predict(M)
    sil.append(silhouette_score(M, labels_k))

# 2. Elegimos el mejor k y reclusterizamos
best_k = range_k[np.argmax(sil)]
labels = AgglomerativeClustering(n_clusters=best_k).fit_predict(M)

# 3. Métrica global de silueta
global_sil = silhouette_score(M, labels)

# 4. Métricas de silueta por muestra y promedio por clúster
sample_sil = silhouette_samples(M, labels)
clusters = {}
for cl in set(labels):
    # términos de este clúster
    terms_in_cl = [t for t, lab in zip(terms, labels) if lab == cl]
    # silueta media del clúster
    sil_cl = sample_sil[labels == cl].mean()
    clusters[cl] = {
        'terms': terms_in_cl,
        'silhouette': sil_cl
    }

# 5. Formateamos para volcar a JSON o documento
clusters_clean = {
    str(cl): {
        'terms': info['terms'],
        'silhouette': float(info['silhouette'])
    }
    for cl, info in clusters.items()
}

result = {
    'best_k': int(best_k),
    'global_silhouette': float(global_sil),
    'clusters': clusters_clean
}

with open(os.path.join(OUTPUT_DIR,'clusters.json'),'w',encoding='utf-8') as f:
    json.dump(clusters_clean, f, ensure_ascii=False, indent=2)

# 4. Generación de explicaciones con Qwen-2.5
# Inicializar modelo
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tok_gen = AutoTokenizer.from_pretrained(gen_model_name)
mod_gen = AutoModelForCausalLM.from_pretrained(gen_model_name).to(torch_device)
gen_pipe = pipeline('text-generation', model=mod_gen, tokenizer=tok_gen, device=0 if torch_device=='cuda' else -1)
explanations = {}

clusters_info = result['clusters']
global_sil = result['global_silhouette']

for cid, info in clusters_info.items():
    cluster_terms = info['terms']
    cluster_sil   = info['silhouette']

    prompt = (

        "Explica en español, de forma clara y concisa, qué representa este cluster, "
        "cómo se relaciona semánticamente con las palabras clave listadas, y qué nos indica la medida de calidad (silhouette) "
        "sobre la cohesión interna y separación frente a otros clusters. "
        "Si la Silhouette es baja, señala posibles problemas de coherencia; si es alta, destaca la solidez semántica del agrupamiento."
        "Devuelve el resultado solamente como un objeto JSON con este formato: {'explicación':'...','coherencia':x,'relevancia':y,'cobertura':z}. "
    
        "La información del cluster y del topic es la siguiente"
        f"— Tópico/cluster: {TOPIC_ID}-{cid}\n"
        f"— Palabras clave del cluster: {', '.join(cluster_terms)}\n"
        f"— Silhouette del cluster: {cluster_sil:.3f} "
        f"(Silhouette global: {global_sil:.3f})\n\n"
    )

    resp = gen_pipe(prompt, max_length=500, do_sample=False)
    explanations['exp-'+str(cid)] = resp[0]['generated_text']
with open(os.path.join(OUTPUT_DIR,'explanations.json'),'w',encoding='utf-8') as f:
    json.dump(explanations, f, ensure_ascii=False, indent=2)

# 5. Evaluación de explicaciones con mT5
# Inicializar modelo de evaluación
tok_eval = T5Tokenizer.from_pretrained(eval_model_name)
mod_eval = T5ForConditionalGeneration.from_pretrained(eval_model_name).to(torch_device)
eval_pipe = pipeline('text2text-generation', model=mod_eval, tokenizer=tok_eval, device=0 if torch_device=='cuda' else -1)
evaluations = {}
print (clusters_clean)
for cid, exp in explanations.items():
    print(exp)
    print (f"Evalúa la explicación para el tópico {TOPIC_ID}-{cid}")
    prompt = (f"Evalúa la explicación para el tópico {TOPIC_ID}-{cid}. "
              f"Términos: {', '.join(clusters_clean[str(cid).replace('exp-','')].get('terms'))}. "
              f"Explicación: {exp}. "
              f"Devuelve sólo un JSON con puntuaciones 1-5 para coherencia, relevancia y cobertura.")
    resp = eval_pipe(prompt, max_length=500)
    evaluations['ev-'+str(cid)] = resp[0]['generated_text']
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
