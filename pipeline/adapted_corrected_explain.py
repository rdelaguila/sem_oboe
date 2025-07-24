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
TOP_K_TERMS        = 10  # número de términos más relevantes por cluster

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
# File: topic_explanation_algorithm.py
"""
Algoritmo de explicabilidad con tripletas, spaCy para similitud,
y clustering jerárquico, generación con Qwen-2.5 y evaluación con mT5.
Configuración en OWN CONFIG.
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
from utils.types import *
from utils.triplet_manager_lib import Tripleta
from operator import itemgetter
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, T5Tokenizer, T5ForConditionalGeneration
import joblib
import nltk
from nltk.corpus import wordnet as wn

# ======== OWN CONFIG ========
TRIPLES_PATH       = 'data/triples_ft/processed/dataset_final_triplet_bbc_pykeen'
TOPIC_ID           = 3
VISITAR_OBJETO     = True
TERMINOS_A_INCLUIR = set(['dvd','google','electronic','tv','sony','screen','nintendo',
                          'player','mobile','phone','software','video','network','apple',
                          'program','linux'])
DBPEDIA_PATH       = 'data/corpus_ft/bbc/diccionario_topic_entidades_dbpedia'
NER_PATH           = 'data/corpus_ft/bbc/diccionario_ner'
N_SINONIMOS        = 1
OUTPUT_DIR         = 'output'
SPACY_MODEL        = 'en_core_web_lg'
GEN_MODEL          = 'Qwen/Qwen2-7B-Instruct'
EVAL_MODEL         = 'google/flan-t5-small'
TOP_K_TERMS        = 10  # número de términos más relevantes por cluster
# ============================

# Preparación
os.makedirs(OUTPUT_DIR, exist_ok=True)
df_tr = joblib.load(TRIPLES_PATH)
topics_dbp = joblib.load(DBPEDIA_PATH)
dictdbp = topics_dbp.get(TOPIC_ID, {})
dictner = joblib.load(NER_PATH)
nlp = spacy.load(SPACY_MODEL)
nltk.download('wordnet', quiet=True)

# Helper funcs
def remove_numbers(text): return re.sub(r"\d+", "", text)

def remove_dbpedia_categories(s): return s.split('/')[-1]

def return_url_element(s):
    for sep in ['#','/']:
        if sep in s:
            s = s.split(sep)[-1]
    return s

# 1. Filtrar tripletas y extraer términos
listado = []
anterior = None
pal_db = set(dictdbp.keys())
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

# 3. Clustering jerárquico + silhouette
labels = 
# Optimización del número de clusters con silhouette
range_n_clusters = range(2, min(len(terms), 50))
valores_medios_silhouette = []

for n_clusters in range_n_clusters:
    modelo = AgglomerativeClustering(
        affinity='euclidean',
        linkage='ward',
        n_clusters=n_clusters
    )
    labels = modelo.fit_predict(M)
    silhouette_avg = silhouette_score(M, labels)
    valores_medios_silhouette.append(silhouette_avg)

# Determina el número óptimo de clusters
optimal_clusters = range_n_clusters[np.argmax(valores_medios_silhouette)]

# Clustering final con el número óptimo
modelo_final = AgglomerativeClustering(
    affinity='euclidean',
    linkage='ward',
    n_clusters=optimal_clusters
)
final_labels = modelo_final.fit_predict(M)

# Plot de silhouette
plt.figure(figsize=(8, 4))
plt.plot(range_n_clusters, valores_medios_silhouette, marker='o')
plt.title('Evolución del índice Silhouette según número de tópicos')
plt.xlabel('Número de tópicos')
plt.ylabel('Índice Silhouette')
plt.savefig(f"{OUTPUT_DIR}/silhouette_evolution.png")
plt.close()

# Dendrograma
plt.figure(figsize=(10, 6))
dendrogram = shc.dendrogram(shc.linkage(M, method='ward'))
plt.title('Dendrograma')
plt.savefig(f"{OUTPUT_DIR}/dendrogram.png")
plt.close()
AgglomerativeClustering(n_clusters=min(len(terms)-1, TOP_K_TERMS)).fit_predict(M)
sample_sil = silhouette_samples(M, labels)
global_sil = silhouette_score(M, labels)
clusters = {}
for cl in set(labels):
    idxs = np.where(labels==cl)[0]
    term_sils = [(terms[i], sample_sil[i]) for i in idxs]
    top_terms = [t for t,_ in sorted(term_sils, key=lambda x: -x[1])[:TOP_K_TERMS]]
    clusters[cl] = top_terms
clusters_str = {str(k): v for k, v in clusters.items()}

with open(os.path.join(OUTPUT_DIR,'clusters.json'),'w', encoding='utf-8') as f:
    json.dump({'best_k': len(clusters_str), 'global_sil': global_sil, 'clusters': clusters_str}, f, ensure_ascii=False, indent=2)

# 4. Generación con Qwen-2.5 + estrategia few-shot + extracción limpia
torch_dev = 'cuda' if torch.cuda.is_available() else 'cpu'
tok_g = AutoTokenizer.from_pretrained(GEN_MODEL)
mod_g = AutoModelForCausalLM.from_pretrained(GEN_MODEL).to(torch_dev)
gen = pipeline('text-generation', model=mod_g, tokenizer=tok_g,
               device=0 if torch.cuda.is_available() else -1,
               return_full_text=False)
# Few-shot examples
ejemplo = (
    "Ejemplo:\n"
    "{'explicación':'Este cluster agrupa términos relacionados con la tecnología móvil y las comunicaciones.',"
    "'coherencia':4,'relevancia':5,'cobertura':4}\n\n"
)
explanations = {}
for cid, terms_c in clusters.items():
    prompt = (
        ejemplo +
        f"Cluster {TOPIC_ID}-{cid}: términos {', '.join(terms_c)}. "
        f"Silhouette global={global_sil:.3f}.\n"
        "Genera solo un JSON con claves 'explicación','coherencia','relevancia','cobertura'."
    )
    out = gen(prompt, max_new_tokens=150, do_sample=False)
    txt = out[0].get('generated_text','')
    try:
        jf = txt[txt.find('{'):txt.rfind('}')+1]
        parsed = json.loads(jf)
    except:
        parsed = {'explicación': txt.strip()}
    explanations[cid] = parsed
    explanations_str = {str(k): v for k, v in explanations.items()}
with open(os.path.join(OUTPUT_DIR,'explanations.json'),'w', encoding='utf-8') as f:
    json.dump(explanations_str, f, ensure_ascii=False, indent=2)

# 5. Evaluación con mT5 + estrategia few-shot avanzado
# Inicializar modelo de evaluación
tok_e = T5Tokenizer.from_pretrained(EVAL_MODEL)
mod_e = T5ForConditionalGeneration.from_pretrained(EVAL_MODEL).to(torch_dev)
evalp = pipeline(
    'text2text-generation', model=mod_e, tokenizer=tok_e,
    device=0 if torch.cuda.is_available() else -1)
# Few-shot para evaluación con justificación breve
ejemplo_eval = (
    "Ejemplo de evaluación con justificación:"
    "{'coherencia':4,'relevancia':5,'cobertura':4}"
    "Justificación: La explicación agrupa bien los términos (coherencia alta), cubre aspectos clave del tópico (relevancia máxima)"
    "y describe adecuadamente la amplitud temática (cobertura alta)."


)
evaluations = {}
for cid, exp in explanations.items():
    terms_c = clusters[cid]
    prompt = (
        ejemplo_eval +
        f"Evalúa esta explicación para el cluster {TOPIC_ID}-{cid}."
        f"Términos clave: {', '.join(terms_c)}."
        f"Explicación: {exp.get('explicación', exp)}."
        "Devuelve un JSON con claves 'coherencia','relevancia','cobertura' y añade un campo 'justificación' con unas 1-2 frases."
    )
    out = evalp(prompt, max_new_tokens=150, do_sample=False)
    txt = out[0].get('generated_text','')
    try:
        jf = txt[txt.find('{'):txt.rfind('}')+1]
        parsed = json.loads(jf)
    except:
        parsed = {'error': txt.strip()}
    evaluations[cid] = parsed

evaluations_str = {str(k): v for k, v in evaluations.items()}

with open(os.path.join(OUTPUT_DIR,'evaluations.json'),'w', encoding='utf-8') as f:
    json.dump(evaluations_str, f, ensure_ascii=False, indent=2)

# 6. Resumen final. Resumen final
with open(os.path.join(OUTPUT_DIR,'summary.txt'),'w', encoding='utf-8') as f:
    f.write(f"Tópico {TOPIC_ID}: {len(clusters)} clusters, silhouette global {global_sil:.3f}\n")
    for cid, terms_c in clusters.items():
        exp = explanations[cid].get('explicación','')
        eva = evaluations[cid]
        coh = eva.get('coherencia','N/A')
        rel = eva.get('relevancia','N/A')
        cov = eva.get('cobertura','N/A')
        f.write(f"Cluster {cid}: explicación='{exp}' coherencia={coh} relevancia={rel} cobertura={cov}\n")
print('Pipeline completo. Revisar output.')

# ======== Advertencia ========
# Revisa manualmente la compatibilidad entre Qwen-2.5 y mT5-small.
# Las salidas de generación deben formatearse claramente para evitar tokens especiales de error.
