#!/usr/bin/env python3
# preprocessing_sequential.py
# Preprocesamiento secuencial optimizado (Mac M1 Max u otros).
# Lee configuración de rutas y jobs desde `config/config.json`.
# Pasos:
#  1) Resolución de coreferencias (spaCy + coreferee)
#  2) Limpieza y normalización de texto (TextAnalyzer)
#   3) Reconocimiento de entidades (Spotlight)
#  4) Enriquecimiento SPARQL + normalización (OntoManager)
# Todo ejecutado de forma secuencial, mostrando tiempos por job y ETA global.

import os
import json
import time
import re
import joblib
import pandas as pd
import spacy
import coreferee
from spotlight import annotate
from collections import defaultdict
import multiprocessing

# Importa tu módulo con TextAnalyzer, SemanticAnalyzer y OntoManager
from preproc_module import TextAnalyzer, SemanticAnalyzer, OntoManager

# --- Utilitarios de coref ---
def resolve_corefs(text: str) -> str:
    doc = resolve_corefs.nlp(text)
    try:
        return doc._.coref_chains.resolve()
    except Exception:
        return text

resolve_corefs.nlp = None

def init_spacy():
    nlp = spacy.load("en_core_web_lg")
    nlp.add_pipe("coreferee")
    return nlp

# --- NER con Spotlight (raw entities) ---
def identify_dbpedia_entities(text: str,
                              endpoint="http://localhost:2222/rest/annotate",
                              confidence=0.5,
                              support=1000) -> dict:
    try:
        annots = annotate(endpoint, text,
                          confidence=confidence,
                          support=support,
                          spotter='Default')
        return {a['surfaceForm']: {'URI': a['URI'], 'types': a.get('types', [])}
                for a in annots}
    except Exception as ex:
        print(f"Spotlight error: {ex}")
        return {}

# --- Pipeline de preprocesamiento ---
def partition_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # 1) Coreferencias
    if resolve_corefs.nlp is None:
        resolve_corefs.nlp = init_spacy()
    df['text_coref'] = df['news'].apply(resolve_corefs)

    # 2) Limpieza y tokenización
    text_analyzer = TextAnalyzer(resolve_corefs.nlp)
    df['text_clean'] = (
        df['text_coref']
          .apply(text_analyzer.remove_special_lines)
          .apply(text_analyzer.strip_formatting)
    )

    # 3) Reconocimiento de entidades DBpedia
    df['entidades_raw'] = df['text_clean'].apply(identify_dbpedia_entities)

    # 4) Enriquecimiento + normalización
    onto_mgr = OntoManager(resolve_corefs.nlp)
    min_interval = 0.2  # s
    last_time = 0

    def _norm_map(ent_map: dict) -> dict:
        nonlocal last_time
        out = {}
        for sf, props in ent_map.items():
            if not props['types']:
                elapsed = time.time() - last_time
                if elapsed < min_interval:
                    time.sleep(min_interval - elapsed)
                uri_q = f"<{props['URI']}>"
                try:
                    props['types'] = onto_mgr._getDBPediaTypes(uri_q)
                except Exception:
                    time.sleep(1)
                    props['types'] = onto_mgr._getDBPediaTypes(uri_q)
                last_time = time.time()
            props['types_normalizados'] = onto_mgr.normalize(props['types'])
            out[sf] = props
        return out

    df['entidades_dbpedia'] = df['entidades_raw'].apply(_norm_map)
    return df

# --- Main secuencial ---
def main():
    # Determina rutas
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    cfg_path    = os.path.join(project_root, 'config', 'config.json')
    with open(cfg_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Normaliza rutas a absolutas
    for key, rel in list(config.items()):
        if not os.path.isabs(rel):
            config[key] = os.path.normpath(os.path.join(project_root, rel))

    # Descubre jobs
    jobs = []
    for key, in_path in config.items():
        if key.endswith("_preproc_input"):
            base     = key[:-len("_preproc_input")]
            out_key  = f"{base}_preproc_output"
            out_path = config.get(out_key)
            if not out_path:
                raise KeyError(f"Falta config para salida {out_key}")
            jobs.append((base, in_path, out_path))

    total      = len(jobs)
    start_all  = time.time()
    durations  = []

    for idx, (name, in_path, out_path) in enumerate(jobs, 1):
        print(f"[{idx}/{total}] Procesando '{name}'...")
        t0 = time.time()

        # Lectura secuencial
        df = pd.read_csv(
            in_path,
            sep=',',
            quotechar='"',
            encoding='latin1',
            on_bad_lines='warn'
        )

        # Preprocesa todo el DataFrame
        processed = partition_preprocess(df)

        # Guarda el resultado
        joblib.dump(processed, out_path)

        dt = time.time() - t0
        durations.append(dt)
        avg = sum(durations) / len(durations)
        eta = (total - idx) * avg
        print(f"✅ '{name}' completado en {dt:.1f}s (ETA: {eta:.1f}s)")

    total_time = time.time() - start_all
    print(f"✔️ Todos los jobs completados en {total_time:.1f}s.")

if __name__ == '__main__':
    main()
