#!/usr/bin/env python3
# preprocessing_parametrizable.py
"""
Semantic Preprocessing Pipeline with Advanced Strategies

Sequential implementation with parameters via command-line or defaults.
Steps:
  1) Coreference Resolution (spaCy + coreferee)
  2) Text Cleaning and Normalization (TextAnalyzer)
  3) spaCy NER Entity Recognition (new step)
  4) Entity Recognition (Spotlight)
  5) SPARQL Enrichment + Normalization (OntoManager)
"""

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

from preproc_module import TextAnalyzer, SemanticAnalyzer, OntoManager


# --- Coreference resolution utilities ---
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


# --- NER with Spotlight (raw entities) ---
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


# --- Preprocessing pipeline ---
def partition_preprocess(df: pd.DataFrame, spotlight_endpoint: str, onto_paths: dict) -> pd.DataFrame:
    # 1) Coreference Resolution
    if resolve_corefs.nlp is None:
        resolve_corefs.nlp = init_spacy()

    print("⏳ Resolving coreference chains...")
    df['text_coref'] = df['text'].apply(resolve_corefs)

    # 2) Text Cleaning and Normalization
    print("⏳ Cleaning and normalizing text...")
    text_analyzer = TextAnalyzer(resolve_corefs.nlp)
    df['text_clean'] = (
        df['text_coref']
        .apply(text_analyzer.remove_special_lines)
        .apply(text_analyzer.strip_formatting)
    )

    # 3) NEW: Entity extraction with spaCy NER (as in original notebooks)
    print("⏳ Extracting entities with spaCy NER...")
    df['entidades'] = df['text_coref'].apply(text_analyzer.extract_spacy_entities)

    # Statistics
    total_entities_spacy = sum(len(ents) for ents in df['entidades'])
    print(f"   ✓ spaCy entities extracted: {total_entities_spacy}")

    # 4) DBpedia Entity Recognition with Spotlight
    print("⏳ Identifying DBpedia entities with Spotlight...")
    df['entidades_raw'] = df['text_clean'].apply(
        lambda text: identify_dbpedia_entities(text, endpoint=spotlight_endpoint)
    )

    # Statistics
    total_entities_dbpedia = sum(len(ents) for ents in df['entidades_raw'])
    print(f"   ✓ DBpedia entities identified: {total_entities_dbpedia}")

    # 5) Enrichment + Normalization
    print("⏳ Enriching entities with semantic types...")
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
            print(f"   → Enriched {entities_processed} entities in this row")
        return out

    df['entidades_dbpedia_simplificadas'] = df['entidades_raw'].apply(_norm_map)

    # Also create column with original format (complete dictionary)
    df['entidades_dbpedia'] = df['entidades_raw']

    print("✓ Preprocessing completed")
    return df


# --- Main parametrizable ---
def main():
    # ---------- COMMAND-LINE ARGUMENTS ----------
    parser = argparse.ArgumentParser(description='Semantic Preprocessing Pipeline')
    parser.add_argument('--input_file', type=str,
                        default='data/corpus_raw/amazon/amazon_corpus_raw.csv',
                        help='Path to input data file')
    parser.add_argument('--output_file', type=str,
                        default='data/processed/amazon/amazon_processed_semantic.pkl',
                        help='Path to output processed file')
    parser.add_argument('--config_file', type=str,
                        default='config/config.json',
                        help='Configuration JSON file')
    parser.add_argument('--spotlight_endpoint', type=str,
                        default='http://localhost:2222/rest/annotate',
                        help='DBpedia Spotlight API endpoint')
    parser.add_argument('--dbo_ontology', type=str,
                        default='data/ontologias/dbpedia_2016-10.owl',
                        help='Path to DBpedia ontology')
    parser.add_argument('--sumo_ontology', type=str,
                        default='data/ontologias/SUMO.owl',
                        help='Path to SUMO ontology')
    parser.add_argument('--input_column', type=str,
                        default='text',
                        help='Text column name to process')
    parser.add_argument('--encoding', type=str,
                        default='latin1',
                        help='Input file encoding')
    parser.add_argument('--separator', type=str,
                        default=';',
                        help='CSV separator')
    parser.add_argument('--batch_size', type=int,
                        default=None,
                        help='Process in batches of N rows (None = all together)')

    # Parse arguments with error handling
    try:
        args = parser.parse_args()
    except SystemExit:
        # If no valid arguments, use defaults
        print("No valid arguments provided. Using default values:")
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

    # ---------- CONFIGURATION ----------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Normalize paths to absolute if relative
    def normalize_path(path):
        if not os.path.isabs(path):
            return os.path.normpath(os.path.join(project_root, path))
        return path

    input_file = normalize_path(args.input_file)
    output_file = normalize_path(args.output_file)
    config_file = normalize_path(args.config_file)
    dbo_path = normalize_path(args.dbo_ontology)
    sumo_path = normalize_path(args.sumo_ontology)

    # Display configuration
    print("=" * 60)
    print("SEMANTIC PREPROCESSING PIPELINE CONFIGURATION")
    print("=" * 60)
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Config file: {config_file}")
    print(f"Spotlight endpoint: {args.spotlight_endpoint}")
    print(f"DBpedia ontology: {dbo_path}")
    print(f"SUMO ontology: {sumo_path}")
    print(f"Text column: {args.input_column}")
    print(f"Encoding: {args.encoding}")
    print(f"Separator: {args.separator}")
    if args.batch_size:
        print(f"Batch size: {args.batch_size}")
    print("=" * 60)

    # Verify input files
    if not os.path.exists(input_file):
        print(f"ERROR: Input file not found: {input_file}")
        print("Please run preprocessing first or verify path.")
        return

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Try loading additional configuration from JSON (optional)
    config_data = {}
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            print(f"✓ Additional configuration loaded from: {config_file}")
        except Exception as e:
            print(f"⚠ Could not load JSON config: {e}")

    # Prepare ontology paths
    onto_paths = {
        'dbo': f"file://{dbo_path}",
        'sumo': f"file://{sumo_path}"
    }

    # Verify ontologies
    for name, path in [('DBpedia', dbo_path), ('SUMO', sumo_path)]:
        if not os.path.exists(path):
            print(f"⚠ WARNING: Ontology not found {name}: {path}")

    # ---------- MAIN PROCESSING ----------
    start_time = time.time()

    try:
        # File reading
        print(f"\n📖 Reading file: {input_file}")
        df = pd.read_csv(
            input_file,
            sep=args.separator,
            quotechar='"',
            encoding=args.encoding,
            on_bad_lines='warn'
        )

        print(f"✓ File read: {len(df)} rows, {len(df.columns)} columns")

        # Verify that text column exists
        if args.input_column not in df.columns:
            print(f"ERROR: Column '{args.input_column}' not found in dataset")
            print(f"Available columns: {list(df.columns)}")
            return
        else:
            df[args.input_column] = df[args.input_column].fillna("")  # Replace NaN with empty string
            df[args.input_column] = df[args.input_column].astype(str)  # Convert everything to string

        # Processing by batches or complete
        if args.batch_size and args.batch_size < len(df):
            print(f"\n⏳ Processing in batches of {args.batch_size} rows...")
            processed_dfs = []

            for i in range(0, len(df), args.batch_size):
                batch_end = min(i + args.batch_size, len(df))
                batch_df = df.iloc[i:batch_end].copy()

                print(f"  Processing batch {i // args.batch_size + 1}: rows {i}-{batch_end - 1}")
                batch_start = time.time()

                processed_batch = partition_preprocess(
                    batch_df,
                    args.spotlight_endpoint,
                    onto_paths
                )
                processed_dfs.append(processed_batch)

                batch_time = time.time() - batch_start
                print(f"  ✓ Batch completed in {batch_time:.1f}s")

            # Combine all batches
            processed = pd.concat(processed_dfs, ignore_index=True)

        else:
            # Complete processing
            print(f"\n⏳ Processing complete dataset...")
            processed = partition_preprocess(df, args.spotlight_endpoint, onto_paths)

        # Save result
        print(f"\n💾 Saving result to: {output_file}")
        joblib.dump(processed, output_file)

        # Final statistics
        total_time = time.time() - start_time
        print(f"\n✓ PROCESSING COMPLETED")
        print(f"⏱ Total time: {total_time:.1f}s ({total_time / 60:.1f} min)")
        print(f"📊 Statistics:")
        print(f"   - Rows processed: {len(processed)}")
        print(f"   - Columns generated: {len(processed.columns)}")

        # Show new columns
        new_cols = [col for col in processed.columns if col not in df.columns]
        if new_cols:
            print(f"   - New columns: {new_cols}")

        # Verify entities found
        if 'entidades' in processed.columns:
            total_spacy_entities = sum(len(entities) for entities in processed['entidades'] if entities)
            print(f"   - Total spaCy entities: {total_spacy_entities}")

        if 'entidades_dbpedia_simplificadas' in processed.columns:
            total_dbpedia_entities = sum(
                len(entities) for entities in processed['entidades_dbpedia_simplificadas'] if entities)
            print(f"   - Total DBpedia entities: {total_dbpedia_entities}")

        print(f"📁 File saved: {output_file}")

    except Exception as e:
        print(f"\nERROR during processing: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 60)
    print("PROCESS COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == '__main__':
    main()