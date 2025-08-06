#!/usr/bin/env python3
# preprocessing_parametrizable.py
# Preprocesamiento secuencial optimizado con parámetros por argumento o defecto.
# Pasos:
#  1) Resolución de coreferencias (spaCy + coreferee)
#  2) Limpieza y normalización de texto (TextAnalyzer)
#  3) Reconocimiento de entidades spaCy NER (nuevo paso)
#  4) Reconocimiento de entidades (Spotlight)
#  5) Enriquecimiento SPARQL + normalización (OntoManager)

import os
import json
import time
import re
import joblib
import pandas as pd
import spacy
import coreferee
import argparse
from spotlight import annotate
from collections import defaultdict

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
def partition_preprocess(df: pd.DataFrame, spotlight_endpoint: str, onto_paths: dict) -> pd.DataFrame:
    # 1) Coreferencias
    if resolve_corefs.nlp is None:
        resolve_corefs.nlp = init_spacy()

    print("🔄 Resolviendo coreferencias...")
    df['text_coref'] = df['text'].apply(resolve_corefs)

    # 2) Limpieza y tokenización
    print("🔄 Limpiando y normalizando texto...")
    text_analyzer = TextAnalyzer(resolve_corefs.nlp)
    df['text_clean'] = (
        df['text_coref']
        .apply(text_analyzer.remove_special_lines)
        .apply(text_analyzer.strip_formatting)
    )

    # 3) NUEVO: Extración de entidades con spaCy NER (como en los notebooks originales)
    print("🔄 Extrayendo entidades con spaCy NER...")
    df['entidades'] = df['text_coref'].apply(text_analyzer.extract_spacy_entities)

    # Estadística
    total_entities_spacy = sum(len(ents) for ents in df['entidades'])
    print(f"   ✓ Entidades spaCy extraídas: {total_entities_spacy}")

    # 4) Reconocimiento de entidades DBpedia con Spotlight
    print("🔄 Identificando entidades DBpedia con Spotlight...")
    df['entidades_raw'] = df['text_clean'].apply(
        lambda text: identify_dbpedia_entities(text, endpoint=spotlight_endpoint)
    )

    # Estadística
    total_entities_dbpedia = sum(len(ents) for ents in df['entidades_raw'])
    print(f"   ✓ Entidades DBpedia identificadas: {total_entities_dbpedia}")

    # 5) Enriquecimiento + normalización
    print("🔄 Enriqueciendo entidades con tipos semánticos...")
    onto_mgr = OntoManager(
        resolve_corefs.nlp,
        path_dbo=onto_paths['dbo'],
        path_sumo=onto_paths['sumo']
    )
    min_interval = 0.2  # s
    last_time = 0

    def _norm_map(ent_map: dict) -> dict:
        nonlocal last_time
        out = {}
        entities_processed = 0

        for sf, props in ent_map.items():
            if not props['types']:
                elapsed = time.time() - last_time
                if elapsed < min_interval:
                    time.sleep(min_interval - elapsed)
                uri_q = f"<{props['URI']}>"
                try:
                    props['types'] = onto_mgr._getDBPediaTypes(uri_q)
                    entities_processed += 1
                except Exception:
                    time.sleep(1)
                    try:
                        props['types'] = onto_mgr._getDBPediaTypes(uri_q)
                        entities_processed += 1
                    except:
                        props['types'] = []
                last_time = time.time()
            props['types_normalizados'] = onto_mgr.normalize(props['types'])
            out[sf] = props

        if entities_processed > 0:
            print(f"   → Enriquecidas {entities_processed} entidades en esta fila")
        return out

    df['entidades_dbpedia_simplificadas'] = df['entidades_raw'].apply(_norm_map)

    # Crear también la columna con el formato original (diccionario completo)
    df['entidades_dbpedia'] = df['entidades_raw']

    print("✅ Preprocesamiento completado")
    return df


# --- Main parametrizable ---
def main():
    # ---------------- ARGUMENTOS DE LÍNEA DE COMANDOS ----------------
    parser = argparse.ArgumentParser(description='Pipeline de Preprocesamiento Semántico')
    parser.add_argument('--input_file', type=str,
                        default='data/corpus_raw/amazon/amazon_corpus_raw.csv',
                        help='Archivo CSV de entrada')
    parser.add_argument('--output_file', type=str,
                        default='data/processed/amazon/amazon_processed_semantic.pkl',
                        help='Archivo de salida procesado')
    parser.add_argument('--config_file', type=str,
                        default='config/config.json',
                        help='Archivo de configuración JSON')
    parser.add_argument('--spotlight_endpoint', type=str,
                        default='http://localhost:2222/rest/annotate',
                        help='Endpoint de DBpedia Spotlight')
    parser.add_argument('--dbo_ontology', type=str,
                        default='data/ontologias/dbpedia_2016-10.owl',
                        help='Ruta a la ontología DBpedia')
    parser.add_argument('--sumo_ontology', type=str,
                        default='data/ontologias/SUMO.owl',
                        help='Ruta a la ontología SUMO')
    parser.add_argument('--input_column', type=str,
                        default='text',
                        help='Nombre de la columna de texto a procesar')
    parser.add_argument('--encoding', type=str,
                        default='latin1',
                        help='Codificación del archivo de entrada')
    parser.add_argument('--separator', type=str,
                        default=';',
                        help='Separador CSV')
    parser.add_argument('--batch_size', type=int,
                        default=None,
                        help='Procesar en lotes de N filas (None = todo junto)')

    # Parsear argumentos con manejo de errores
    try:
        args = parser.parse_args()
    except SystemExit:
        # Si no hay argumentos válidos, usar valores por defecto
        print("No se proporcionaron argumentos válidos. Usando valores por defecto:")
        args = argparse.Namespace(
            input_file='data/corpus_raw/amazon/amazon_corpus_raw.csv',
            output_file='data/processed/amazon/amazon_processed_semantic.pkl',
            config_file='config/config.json',
            spotlight_endpoint='http://localhost:2222/rest/annotate',
            dbo_ontology='data/ontologias/dbpedia_2016-10.owl',
            sumo_ontology='data/ontologias/SUMO.owl',
            input_column='text',
            encoding='latin1',
            separator=';',
            batch_size=None
        )

    # ---------------- CONFIGURACIÓN ----------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Normalizar rutas a absolutas si son relativas
    def normalize_path(path):
        if not os.path.isabs(path):
            return os.path.normpath(os.path.join(project_root, path))
        return path

    input_file = normalize_path(args.input_file)
    output_file = normalize_path(args.output_file)
    config_file = normalize_path(args.config_file)
    dbo_path = normalize_path(args.dbo_ontology)
    sumo_path = normalize_path(args.sumo_ontology)

    # Mostrar configuración
    print("=" * 60)
    print("CONFIGURACIÓN DEL PIPELINE DE PREPROCESAMIENTO")
    print("=" * 60)
    print(f"Archivo de entrada: {input_file}")
    print(f"Archivo de salida: {output_file}")
    print(f"Archivo de config: {config_file}")
    print(f"Endpoint Spotlight: {args.spotlight_endpoint}")
    print(f"Ontología DBpedia: {dbo_path}")
    print(f"Ontología SUMO: {sumo_path}")
    print(f"Columna de texto: {args.input_column}")
    print(f"Codificación: {args.encoding}")
    print(f"Separador: {args.separator}")
    if args.batch_size:
        print(f"Tamaño de lote: {args.batch_size}")
    print("=" * 60)

    # Verificar archivos de entrada
    if not os.path.exists(input_file):
        print(f"ERROR: No se encontró el archivo de entrada: {input_file}")
        return

    # Crear directorio de salida si no existe
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Intentar cargar configuración adicional desde JSON (opcional)
    config_data = {}
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            print(f"✓ Configuración adicional cargada desde: {config_file}")
        except Exception as e:
            print(f"⚠ No se pudo cargar config JSON: {e}")

    # Preparar rutas de ontologías
    onto_paths = {
        'dbo': f"file://{dbo_path}",
        'sumo': f"file://{sumo_path}"
    }

    # Verificar ontologías
    for name, path in [('DBpedia', dbo_path), ('SUMO', sumo_path)]:
        if not os.path.exists(path):
            print(f"⚠ ADVERTENCIA: No se encontró ontología {name}: {path}")

    # ---------------- PROCESAMIENTO PRINCIPAL ----------------
    start_time = time.time()

    try:
        # Lectura del archivo
        print(f"\n📖 Leyendo archivo: {input_file}")
        df = pd.read_csv(
            input_file,
            sep=args.separator,
            quotechar='"',
            encoding=args.encoding,
            on_bad_lines='warn'
        )

        print(f"✓ Archivo leído: {len(df)} filas, {len(df.columns)} columnas")

        # Verificar que existe la columna de texto
        if args.input_column not in df.columns:
            print(f"ERROR: Columna '{args.input_column}' no encontrada en el dataset")
            print(f"Columnas disponibles: {list(df.columns)}")
            return
        else:
            df[args.input_column] = df[args.input_column].fillna("")  # Reemplazar NaN con cadena vacía
            df[args.input_column] = df[args.input_column].astype(str)  # Convertir todo a string

        # Procesamiento por lotes o completo
        if args.batch_size and args.batch_size < len(df):
            print(f"\n🔄 Procesando en lotes de {args.batch_size} filas...")
            processed_dfs = []

            for i in range(0, len(df), args.batch_size):
                batch_end = min(i + args.batch_size, len(df))
                batch_df = df.iloc[i:batch_end].copy()

                print(f"  Procesando lote {i // args.batch_size + 1}: filas {i}-{batch_end - 1}")
                batch_start = time.time()

                processed_batch = partition_preprocess(
                    batch_df,
                    args.spotlight_endpoint,
                    onto_paths
                )
                processed_dfs.append(processed_batch)

                batch_time = time.time() - batch_start
                print(f"  ✓ Lote completado en {batch_time:.1f}s")

            # Combinar todos los lotes
            processed = pd.concat(processed_dfs, ignore_index=True)

        else:
            # Procesamiento completo
            print(f"\n🔄 Procesando dataset completo...")
            processed = partition_preprocess(df, args.spotlight_endpoint, onto_paths)

        # Guardar resultado
        print(f"\n💾 Guardando resultado en: {output_file}")
        joblib.dump(processed, output_file)

        # Estadísticas finales
        total_time = time.time() - start_time
        print(f"\n✅ PROCESAMIENTO COMPLETADO")
        print(f"⏱️  Tiempo total: {total_time:.1f}s ({total_time / 60:.1f} min)")
        print(f"📊 Estadísticas:")
        print(f"   - Filas procesadas: {len(processed)}")
        print(f"   - Columnas generadas: {len(processed.columns)}")

        # Mostrar columnas nuevas
        new_cols = [col for col in processed.columns if col not in df.columns]
        if new_cols:
            print(f"   - Nuevas columnas: {new_cols}")

        # Verificar entidades encontradas
        if 'entidades' in processed.columns:
            total_spacy_entities = sum(len(entities) for entities in processed['entidades'] if entities)
            print(f"   - Total entidades spaCy: {total_spacy_entities}")

        if 'entidades_dbpedia_simplificadas' in processed.columns:
            total_dbpedia_entities = sum(
                len(entities) for entities in processed['entidades_dbpedia_simplificadas'] if entities)
            print(f"   - Total entidades DBpedia: {total_dbpedia_entities}")

        print(f"📁 Archivo guardado: {output_file}")

    except Exception as e:
        print(f"\n❌ ERROR durante el procesamiento: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 60)
    print("PROCESO COMPLETADO EXITOSAMENTE")
    print("=" * 60)


if __name__ == '__main__':
    main()