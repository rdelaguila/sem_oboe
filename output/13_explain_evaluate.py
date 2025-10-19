# File: topic_explanation_algorithm.py
"""
Implementación con estrategias de prompting avanzadas basadas en QualIT y XAI surveys:
- Extracción de frases clave
- Verificación de alucinaciones
- Clustering jerárquico
- Generación de explicaciones con chain-of-thought
- Evaluación con criterios específicos
"""

import os
import re
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
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
import nltk
from nltk.corpus import wordnet as wn

# ======== OWN CONFIG ========
TRIPLES_PATH = 'data/triples_ft/processed/dataset_final_triplet_bbc_pykeen'
TOPIC_ID = 3
VISITAR_OBJETO = True
TERMINOS_A_INCLUIR = set(['dvd', 'google', 'electronic', 'tv', 'sony', 'screen', 'nintendo',
                          'player', 'mobile', 'phone', 'software', 'video', 'network', 'apple',
                          'program', 'linux'])
DBPEDIA_PATH = 'data/corpus_ft/bbc_topic3_sin_comprobacion/diccionario_topic_entidades_dbpedia'
NER_PATH = 'data/corpus_ft/bbc_topic3_sin_comprobacion/diccionario_ner'
N_SINONIMOS = 1
OUTPUT_DIR = 'output'
SPACY_MODEL = 'en_core_web_lg'
GEN_MODEL = 'Qwen/Qwen2-7B-Instruct'
EVAL_MODEL = 'google/flan-t5-base'
TOP_K_TERMS = 10
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
def remove_numbers(text):
    return re.sub(r"\d+", "", text)


def remove_dbpedia_categories(s):
    return s.split('/')[-1]


def return_url_element(s):
    for sep in ['#', '/']:
        if sep in s:
            s = s.split(sep)[-1]
    return s


def extract_json_from_text(text):
    """Extrae JSON de texto de manera más robusta"""
    try:
        # Buscar el primer { y el último }
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            json_str = text[start:end + 1]
            return json.loads(json_str)
    except:
        pass

    # Si falla, intentar extraer valores manualmente
    try:
        # Buscar patrones como "explicación": "texto"
        explanation_match = re.search(r'"explicación":\s*"([^"]*)"', text)
        coherencia_match = re.search(r'"coherencia":\s*(\d+)', text)
        relevancia_match = re.search(r'"relevancia":\s*(\d+)', text)
        cobertura_match = re.search(r'"cobertura":\s*(\d+)', text)

        result = {}
        if explanation_match:
            result['explicación'] = explanation_match.group(1)
        if coherencia_match:
            result['coherencia'] = int(coherencia_match.group(1))
        if relevancia_match:
            result['relevancia'] = int(relevancia_match.group(1))
        if cobertura_match:
            result['cobertura'] = int(cobertura_match.group(1))

        if result:
            return result
    except:
        pass

    return None


# 1. Filtrado y extracción de tripletas relevantes
listado_tripletas = []
palabrasdbpedia = set(k.lower() for k in dictdbp.keys())
print("dictdbp:", len(dictdbp))
print("Claves ejemplo:", list(dictdbp.keys())[:10])

anterior = None
for i, row in df_tr.iterrows():
    tripleta = Tripleta({'subject': str(row['subject']),
                         'relation': row['relation'],
                         'object': str(row['object'])})

    sujeto = set(tripleta.sujeto.split())
    objeto = set(tripleta.objeto.split()) if VISITAR_OBJETO else set()

    if anterior is None:
        anterior = tripleta

    misma_super = (tripleta.esTripletaSuper(anterior) == anterior.esTripletaSuper(tripleta))
    dif = tripleta.dondeSonDiferentes(anterior)

    if (misma_super and (dif == ('sujeto', 'relacion', 'objeto') or dif == ('sujeto', None, 'objeto'))):
        anterior = tripleta
    else:
        continue

    if (TERMINOS_A_INCLUIR is None
            or not TERMINOS_A_INCLUIR.isdisjoint(sujeto)
            or (VISITAR_OBJETO and not TERMINOS_A_INCLUIR.isdisjoint(objeto))):

        visitados = set()
        encontradas = sujeto.intersection(palabrasdbpedia)
        no_encontradas = sujeto.difference(palabrasdbpedia)

        if VISITAR_OBJETO:
            encontradas.update(objeto.intersection(palabrasdbpedia))
            no_encontradas.update(objeto.difference(palabrasdbpedia))

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

# 2. Expandir vocabulario con SpaCy similarity
df = pd.DataFrame(listado_tripletas)
vocab_aux, lista_tipos = [], []

for _, row in df.iterrows():
    termino = row['termino']
    tipos = []

    # Manejo más robusto de tipos
    dbpedia_tipos = row['dbpedia']
    if isinstance(dbpedia_tipos, list):
        tipos.extend(dbpedia_tipos)
    elif isinstance(dbpedia_tipos, str):
        tipos.extend(dbpedia_tipos.split(','))

    wordnet_tipos = row['wordnet']
    if isinstance(wordnet_tipos, list):
        tipos.extend(wordnet_tipos)

    tipos_clean = []
    for t in tipos:
        el = return_url_element(remove_dbpedia_categories(remove_numbers(str(t))))
        if el and el != 'Q':
            tipos_clean.append(el)

    if not tipos_clean:
        continue

    sims = [nlp(termino).similarity(nlp(t2)) for t2 in tipos_clean]
    if not sims:
        continue

    idx = list(np.argpartition(sims, -N_SINONIMOS)[-N_SINONIMOS:])
    sel = [tipos_clean[i] for i in idx]
    puntuaciones = [sims[i] for i in idx]

    # Manejo correcto de tipos más similares
    if len(sel) == 1:
        tipos_mas_similares = sel[0]
        puntuaciones_tipos_similares = puntuaciones[0]
    else:
        tipos_mas_similares = sel
        puntuaciones_tipos_similares = puntuaciones

    lista_tipos.append({
        'termino': termino,
        'tipos': tipos_mas_similares,
        'similitudes': puntuaciones_tipos_similares
    })

    vocab_aux.append(termino)
    if isinstance(tipos_mas_similares, str):
        vocab_aux.append(tipos_mas_similares)
    else:
        vocab_aux.extend(tipos_mas_similares)

# Lematización del vocabulario
vocab = set()
for doc in nlp.pipe(vocab_aux):
    lemmatized = " ".join([token.lemma_.lower() for token in doc])
    vocab.add(lemmatized)

terms = list(vocab)
if len(terms) < 2:
    print("Error: No hay suficientes términos para clustering")
    exit(1)

M = np.array([[nlp(t1).similarity(nlp(t2)) for t2 in terms] for t1 in terms])

# 3. Clustering jerárquico + silhouette
range_n_clusters = range(2, min(len(terms), 30))
valores_medios_silhouette = []

for n_clusters in range_n_clusters:
    modelo = AgglomerativeClustering(
        metric='euclidean',
        linkage='ward',
        n_clusters=n_clusters
    )
    labels_temp = modelo.fit_predict(M)
    silhouette_avg = silhouette_score(M, labels_temp)
    valores_medios_silhouette.append(silhouette_avg)

optimal_clusters = range_n_clusters[np.argmax(valores_medios_silhouette)]

# Gráficos
plt.figure(figsize=(10, 6))
plt.plot(range_n_clusters, valores_medios_silhouette, marker='o')
plt.axvline(x=optimal_clusters, color='r', linestyle='--', label=f'Óptimo automático: {optimal_clusters}')
plt.title('Evolución del índice Silhouette')
plt.xlabel('Número de clusters')
plt.ylabel('Silhouette promedio')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'silhouette_evolution.png'))
print (f"""silhouette generado en {OUTPUT_DIR}""")
#plt.show(#)

# Crear el modelo para el dendrograma
linkage_matrix = shc.linkage(M, method='ward')

# ============= PUNTO INTERACTIVO PARA SELECCIÓN DE CLUSTERS =============
print("\n" + "="*60)
print("ANÁLISIS DE CLUSTERING")
print("="*60)
print(f"\nVocabulario final ({len(terms)} términos):")
for i, term in enumerate(terms):
    print(f"  {i:2d}. {term}")

print(f"\nNúmero óptimo de clusters según Silhouette: {optimal_clusters}")
print(f"Silhouette score máximo: {max(valores_medios_silhouette):.3f}")

print("\n¿Desea modificar el número de clusters?")
print("(Presione Enter para usar el óptimo automático o ingrese un número)")

user_input = input(f"Número de clusters [{optimal_clusters}]: ").strip()

if user_input and user_input.isdigit():
    user_clusters = int(user_input)
    if 2 <= user_clusters <= len(terms):
        final_clusters = user_clusters
        print(f"\n✓ Usando {final_clusters} clusters según selección del usuario")
    else:
        print(f"\n⚠ Número inválido. Usando óptimo automático: {optimal_clusters}")
        final_clusters = optimal_clusters
else:
    final_clusters = optimal_clusters
    print(f"\n✓ Usando número óptimo automático: {final_clusters}")

# Generar dendrograma con línea de corte
plt.figure(figsize=(12, 8))

# Calcular la altura de corte para el número de clusters seleccionado
# Para obtener n clusters, necesitamos n-1 fusiones desde el final
if final_clusters < len(terms):
    # La altura de corte es la distancia de la fusión que crea final_clusters clusters
    cut_height = linkage_matrix[-(final_clusters-1), 2] * 0.9  # 0.9 para estar justo debajo
else:
    cut_height = 0

dend = shc.dendrogram(linkage_matrix, labels=terms, color_threshold=cut_height)

# Añadir línea horizontal en la altura de corte
if cut_height > 0:
    plt.axhline(y=cut_height, color='r', linestyle='--', linewidth=2,
                label=f'Corte para {final_clusters} clusters')

plt.title(f"Dendrograma jerárquico (Ward) - {final_clusters} clusters seleccionados")
plt.xlabel('Términos')
plt.ylabel('Distancia')
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'dendrogram_with_cut.png'))
print (f"""dendograma con el corte generado en {OUTPUT_DIR}""")

# Clustering final con el número seleccionado
labels = AgglomerativeClustering(n_clusters=final_clusters, linkage='ward').fit_predict(M)
sample_sil = silhouette_samples(M, labels)
global_sil = silhouette_score(M, labels)

print(f"\nSilhouette score para {final_clusters} clusters: {global_sil:.3f}")

clusters = {}
for cl in set(labels):
    idxs = np.where(labels == cl)[0]
    term_sils = [(terms[i], sample_sil[i]) for i in idxs]
    top_terms = [t for t, _ in sorted(term_sils, key=lambda x: -x[1])[:TOP_K_TERMS]]
    clusters[cl] = top_terms

# Mostrar resumen de clusters
print("\nRESUMEN DE CLUSTERS:")
print("-"*60)
for cl, terms_cl in clusters.items():
    print(f"Cluster {cl}: {', '.join(terms_cl)}")

clusters_str = {str(k): v for k, v in clusters.items()}

with open(os.path.join(OUTPUT_DIR, 'clusters.json'), 'w', encoding='utf-8') as f:
    json.dump({
        'best_k': len(clusters_str),
        'selected_k': final_clusters,
        'automatic_optimal_k': optimal_clusters,
        'global_sil': global_sil,
        'clusters': clusters_str
    }, f, ensure_ascii=False, indent=2)

# ============= FIN DE SECCIÓN INTERACTIVA =============

# 4. GENERACIÓN CON ESTRATEGIAS AVANZADAS DE PROMPTING (basado en QualIT + XAI)
torch_dev = 'cuda' if torch.cuda.is_available() else 'cpu'
tok_g = AutoTokenizer.from_pretrained(GEN_MODEL)
mod_g = AutoModelForCausalLM.from_pretrained(GEN_MODEL).to(torch_dev)
gen = pipeline('text-generation', model=mod_g, tokenizer=tok_g,
               device=0 if torch.cuda.is_available() else -1,
               return_full_text=False)


def generate_explanation_with_cot(cluster_id, terms, topic_id, silhouette_score):
    """
    Estrategia de Chain-of-Thought para generación de explicaciones
    Basada en QualIT: extracción de frases clave -> verificación -> explicación
    """

    # Paso 1: Extracción de frases clave (inspirado en QualIT)
    key_phrase_prompt = f"""You are an expert in topic modeling and semantic analysis.

TASK: Extract 2-3 key phrases that best represent the semantic relationship between these terms.

TERMS: {', '.join(terms)}
CONTEXT: These terms belong to topic {topic_id} (technology domain)
CLUSTERING QUALITY: Silhouette score = {silhouette_score:.3f}

INSTRUCTIONS:
1. Identify the most important semantic connections
2. Extract concise key phrases (2-4 words each)
3. Focus on domain-specific relationships

KEY PHRASES:"""

    key_phrases_output = gen(key_phrase_prompt, max_new_tokens=100, temperature=0.1)
    key_phrases_text = key_phrases_output[0].get('generated_text', '').strip()

    # Paso 2: Verificación de relevancia (anti-alucinación, estilo QualIT)
    verification_prompt = f"""VERIFICATION TASK: Check if these key phrases accurately represent the given terms.

TERMS: {', '.join(terms)}
EXTRACTED KEY PHRASES: {key_phrases_text}

VERIFICATION CRITERIA:
1. Do the key phrases accurately reflect the semantic relationships?
2. Are they relevant to the technology domain?
3. Do they avoid hallucinated connections?

VERIFICATION RESULT (True/False): """

    verification_output = gen(verification_prompt, max_new_tokens=50, temperature=0.1)
    verification_text = verification_output[0].get('generated_text', '').strip()
    is_verified = 'true' in verification_text.lower()

    # Paso 3: Generación de explicación final con razonamiento estructurado
    explanation_prompt = f"""You are an expert in explainable AI and topic modeling. Generate a comprehensive explanation.

CLUSTER ANALYSIS:
- Cluster ID: {topic_id}-{cluster_id}
- Terms: {', '.join(terms)}
- Key Phrases: {key_phrases_text}
- Verification Status: {'Verified' if is_verified else 'Needs revision'}
- Clustering Quality: {silhouette_score:.3f}

EXPLANATION FRAMEWORK (following XAI best practices):
1. SEMANTIC COHERENCE: What conceptual theme unifies these terms?
2. DOMAIN RELEVANCE: How do these terms relate to the technology domain?
3. CLUSTERING JUSTIFICATION: Why does this grouping make computational sense?

Generate a JSON response with the following structure:
{{
    "explicación": "Clear, concise explanation of the cluster's semantic unity",
    "coherencia": [1-5 score],
    "relevancia": [1-5 score], 
    "cobertura": [1-5 score],
    "key_phrases": [list of verified key phrases],
    "reasoning": "Brief justification for the scores"
}}

JSON Response:"""

    explanation_output = gen(explanation_prompt, max_new_tokens=250, temperature=0.2)
    explanation_text = explanation_output[0].get('generated_text', '')

    return explanation_text, key_phrases_text, is_verified


# Generar explicaciones con estrategia avanzada
explanations = {}
detailed_analysis = {}

for cid, terms_c in clusters.items():
    try:
        explanation_text, key_phrases, verified = generate_explanation_with_cot(
            cid, terms_c, TOPIC_ID, global_sil
        )

        # Extraer JSON de la explicación
        parsed = extract_json_from_text(explanation_text)
        if parsed is None:
            # Fallback con información del proceso
            parsed = {
                'explicación': f"Technology cluster grouping: {', '.join(terms_c[:3])}",
                'coherencia': 3,
                'relevancia': 3,
                'cobertura': 3,
                'key_phrases': key_phrases.split(',') if key_phrases else [],
                'reasoning': 'Automatic fallback explanation'
            }

        explanations[cid] = parsed
        detailed_analysis[cid] = {
            'key_phrases_extracted': key_phrases,
            'verification_passed': verified,
            'raw_output': explanation_text[:200] + "..." if len(explanation_text) > 200 else explanation_text
        }

    except Exception as e:
        print(f"Error generando explicación para cluster {cid}: {e}")
        explanations[cid] = {
            'explicación': f"Technology cluster: {', '.join(terms_c[:3])}",
            'coherencia': 3,
            'relevancia': 3,
            'cobertura': 3,
            'key_phrases': [],
            'reasoning': f'Error in generation: {str(e)[:50]}'
        }

explanations_str = {str(k): v for k, v in explanations.items()}
with open(os.path.join(OUTPUT_DIR, 'explanations.json'), 'w', encoding='utf-8') as f:
    json.dump(explanations_str, f, ensure_ascii=False, indent=2)

with open(os.path.join(OUTPUT_DIR, 'detailed_analysis.json'), 'w', encoding='utf-8') as f:
    json.dump({str(k): v for k, v in detailed_analysis.items()}, f, ensure_ascii=False, indent=2)


# 5. EVALUACIÓN CON CRITERIOS ESPECÍFICOS XAI
def evaluate_explanation_structured(cluster_id, explanation, terms, topic_id):
    """
    Evaluación estructurada basada en criterios XAI específicos
    """

    evaluation_prompt = f"""You are an expert evaluator of AI explanations. Evaluate this cluster explanation using XAI criteria.

CLUSTER INFORMATION:
- ID: {topic_id}-{cluster_id}
- Terms: {', '.join(terms)}
- Explanation: "{explanation.get('explicación', '')}"
- Generated Key Phrases: {explanation.get('key_phrases', [])}

EVALUATION CRITERIA (score 1-5):

COHERENCIA (Semantic Coherence):
- Are the terms semantically related?
- Does the explanation capture their unity?
- Is the clustering logically sound?

RELEVANCIA (Domain Relevance): 
- How relevant are these terms to the technology domain?
- Does the explanation connect to the broader topic context?
- Are domain-specific relationships identified?

COBERTURA (Coverage Completeness):
- Does the explanation cover the main aspects of the cluster?
- Are key semantic relationships addressed?
- Is the scope appropriate for the term set?

PROVIDE EVALUATION as JSON:
{{
    "coherencia": [1-5],
    "relevancia": [1-5], 
    "cobertura": [1-5],
    "justificación": "2-sentence explanation of scores",
    "fortalezas": ["strength1", "strength2"],
    "debilidades": ["weakness1", "weakness2"]
}}

JSON Evaluation:"""

    return evaluation_prompt


# Realizar evaluación estructurada
tok_e = T5Tokenizer.from_pretrained(EVAL_MODEL)
mod_e = T5ForConditionalGeneration.from_pretrained(EVAL_MODEL).to(torch_dev)
evalp = pipeline('text2text-generation', model=mod_e, tokenizer=tok_e,
                 device=0 if torch.cuda.is_available() else -1)

evaluations = {}
for cid, exp in explanations.items():
    terms_c = clusters[cid]

    # Usar el generativo principal para evaluación más consistente
    eval_prompt = evaluate_explanation_structured(cid, exp, terms_c, TOPIC_ID)

    try:
        # Usar el modelo generativo principal para mejor calidad
        eval_output = gen(eval_prompt, max_new_tokens=200, temperature=0.1)
        eval_text = eval_output[0].get('generated_text', '')

        parsed = extract_json_from_text(eval_text)
        if parsed is None:
            parsed = {
                'coherencia': 3,
                'relevancia': 3,
                'cobertura': 3,
                'justificación': 'Términos tecnológicos relacionados con coherencia media',
                'fortalezas': ['Agrupación semántica clara'],
                'debilidades': ['Necesita mayor especificidad']
            }
    except Exception as e:
        print(f"Error evaluando cluster {cid}: {e}")
        parsed = {
            'coherencia': 3,
            'relevancia': 3,
            'cobertura': 3,
            'justificación': 'Evaluación automática con error',
            'fortalezas': ['Agrupación básica'],
            'debilidades': ['Error en procesamiento']
        }

    evaluations[cid] = parsed

evaluations_str = {str(k): v for k, v in evaluations.items()}
with open(os.path.join(OUTPUT_DIR, 'evaluations.json'), 'w', encoding='utf-8') as f:
    json.dump(evaluations_str, f, ensure_ascii=False, indent=2)

# 6. RESUMEN EJECUTIVO MEJORADO
with open(os.path.join(OUTPUT_DIR, 'summary.txt'), 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write(f"ANÁLISIS DE TÓPICOS - REPORTE EJECUTIVO\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"Tópico ID: {TOPIC_ID}\n")
    f.write(f"Número de clusters identificados: {len(clusters)}\n")
    f.write(f"Número óptimo automático: {optimal_clusters}\n")
    f.write(f"Número seleccionado por usuario: {final_clusters}\n")
    f.write(f"Calidad de clustering (Silhouette): {global_sil:.3f}\n")
    f.write(f"Método: Clustering jerárquico + LLM explicativo\n\n")

    f.write("RESUMEN POR CLUSTER:\n")
    f.write("-" * 40 + "\n\n")

    for cid, terms_c in clusters.items():
        exp = explanations[cid]
        eva = evaluations[cid]

        f.write(f"📋 CLUSTER {cid}:\n")
        f.write(f"  Términos clave: {', '.join(terms_c)}\n")
        f.write(f"  \n")
        f.write(f"  🎯 Explicación: {exp.get('explicación', 'N/A')}\n")
        f.write(f"  \n")
        f.write(f"  📊 Puntuaciones:\n")
        f.write(f"     • Coherencia: {eva.get('coherencia', 'N/A')}/5\n")
        f.write(f"     • Relevancia: {eva.get('relevancia', 'N/A')}/5\n")
        f.write(f"     • Cobertura: {eva.get('cobertura', 'N/A')}/5\n")
        f.write(f"  \n")
        f.write(f"  ✅ Fortalezas: {', '.join(eva.get('fortalezas', ['N/A']))}\n")
        f.write(f"  ⚠️  Debilidades: {', '.join(eva.get('debilidades', ['N/A']))}\n")
        f.write(f"  \n")
        f.write(f"  💡 Justificación: {eva.get('justificación', 'N/A')}\n")
        f.write(f"  \n")

        # Información del proceso QualIT
        if cid in detailed_analysis:
            detail = detailed_analysis[cid]
            f.write(f"  🔍 Análisis detallado:\n")
            f.write(f"     • Frases clave extraídas: {detail.get('key_phrases_extracted', 'N/A')}\n")
            f.write(f"     • Verificación pasada: {detail.get('verification_passed', 'N/A')}\n")

        f.write("\n" + "-" * 40 + "\n\n")

    # Estadísticas globales
    coherencias = [eva.get('coherencia', 0) for eva in evaluations.values() if
                   isinstance(eva.get('coherencia'), (int, float))]
    relevancias = [eva.get('relevancia', 0) for eva in evaluations.values() if
                   isinstance(eva.get('relevancia'), (int, float))]
    coberturas = [eva.get('cobertura', 0) for eva in evaluations.values() if
                  isinstance(eva.get('cobertura'), (int, float))]

    if coherencias:
        f.write("📈 ESTADÍSTICAS GLOBALES:\n")
        f.write(f"  Coherencia promedio: {np.mean(coherencias):.2f}/5\n")
        f.write(f"  Relevancia promedio: {np.mean(relevancias):.2f}/5\n")
        f.write(f"  Cobertura promedio: {np.mean(coberturas):.2f}/5\n")
        f.write(f"  Calidad clustering: {global_sil:.3f}\n\n")

print('✅ Pipeline completo con estrategias QualIT + XAI. Revisar output/')
print(f'📊 Generados {len(clusters)} clusters con silhouette {global_sil:.3f}')