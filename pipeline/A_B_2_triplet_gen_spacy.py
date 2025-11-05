#!/usr/bin/env python
# coding: utf-8

"""
VERSIÓN OPTIMIZADA con spaCy
- Reemplaza CoreNLP/Stanza por spaCy (10x más rápido)
- Mantiene TODA la lógica de patrones y validación
- Resto de funciones IGUALES
"""

import argparse
import os
import pandas as pd
import spacy
import joblib
import dask.dataframe as dd
import traceback
from time import sleep

# Importar las nuevas clases de spaCy
from utils.triplet_manager_lib_spacy import (
    TripletGeneratorSpacy,
    ValidadorTripletasSpacy,
    return_triplets_spacy
)


# ============================================================================
# FUNCIÓN MODIFICADA (única diferencia)
# ============================================================================

def annotate_row_spacy(row, nlp, triplet_generator, validator,
                       enable_pos_filtering=True, debug_rejected=False):
    """
    REEMPLAZO de annotate_row_optimized
    Usa spaCy en lugar de CoreNLP
    Resto de lógica IDÉNTICA
    """
    result = []

    # Get topic information
    topic = row.get('topic')

    # Get text - IGUAL que antes
    text_column = ''
    for text_col in ['text_coref', 'coref_text', 'text', 'news']:
        if text_col in row and row[text_col]:
            text_column = str(row[text_col])
            break

    if not text_column:
        return result

    # Process each phrase - IGUAL que antes
    for phrase in text_column.split('.'):
        if not phrase.strip():
            continue

        try:
            # ========== ÚNICO CAMBIO: spaCy en vez de CoreNLP ==========
            doc = nlp(phrase)

            # Extraer triplets (mantiene tu lógica)
            t = return_triplets_spacy(doc, triplet_generator)

            # Crear diccionario POS (compatible con validador)
            d = {token.text: token.tag_ for token in doc}
            # ============================================================

            # Validación - IGUAL que antes
            if enable_pos_filtering:
                valid_triplets = []
                for triplet in t:
                    if debug_rejected:
                        is_valid, explanation = validator.quick_validation_with_pos(
                            triplet, d, debug=True
                        )
                        if is_valid:
                            valid_triplets.append(triplet)
                            print(f"✓ ACCEPTED: {triplet.get('subject')} | "
                                  f"{triplet.get('relation')} | {triplet.get('object')}")
                            print(f"    Reason: {explanation}")
                        else:
                            print(f"✗ REJECTED: {triplet.get('subject')} | "
                                  f"{triplet.get('relation')} | {triplet.get('object')}")
                            print(f"    Reason: {explanation}")
                    else:
                        if validator.quick_validation_with_pos(triplet, d):
                            valid_triplets.append(triplet)

                result.extend(valid_triplets)
            else:
                result.extend(t)

        except Exception as e:
            print(f"Error processing phrase '{phrase[:50]}...': {e}")
            continue

    return result


def transform_partition_spacy(df, nlp, triplet_generator, validator,
                              enable_pos_filtering, debug_rejected):
    """
    REEMPLAZO de transform_partition_optimized
    Pasa objetos de spaCy en lugar de CoreNLP client
    """
    return df.apply(
        lambda row: annotate_row_spacy(
            row, nlp, triplet_generator, validator,
            enable_pos_filtering, debug_rejected
        ),
        axis=1
    )


# ============================================================================
# FUNCIONES ORIGINALES SIN CAMBIOS
# ============================================================================

def deserialize_triplets(triplet_str):
    """Convert triplet string to actual list - SIN CAMBIOS"""
    if pd.isna(triplet_str) or triplet_str == '' or triplet_str == '[]':
        return []

    try:
        import ast
        return ast.literal_eval(triplet_str)
    except:
        try:
            return eval(triplet_str)
        except:
            print(f"WARNING: Could not deserialize: {triplet_str[:100]}...")
            return []


def create_triplets_csv(df, output_dir, csv_filename, include_topics=True):
    """
    Helper function to create triplets CSV - SIN CAMBIOS
    """
    print(f"\nTransforming to triplets dataframe...")

    try:
        # Remove rows without triplets
        original_len = len(df)
        df_filtered = df[df['triplets'].map(len) > 0].copy()
        filtered_len = len(df_filtered)
        print(f"Rows filtered: {original_len} -> {filtered_len} "
              f"({original_len - filtered_len} without triplets)")

        if filtered_len == 0:
            print("WARNING: No triplets to process")
            return

        # Explode triplets
        df_filtered['triplets'] = df_filtered['triplets'].apply(deserialize_triplets)
        triplet_df_def = df_filtered.explode('triplets', ignore_index=True)

        # Filter possible None values
        triplet_df_def = triplet_df_def[triplet_df_def['triplets'].notna()]
        triplet_df_kge = pd.DataFrame(
            triplet_df_def['triplets'].tolist(),
            index=triplet_df_def.index
        )

        # Add metadata
        topic_col_found = None
        if include_topics:
            for topic_col in ['topic', 'new_target', 'target']:
                if topic_col in triplet_df_def.columns:
                    triplet_df_kge['new_topic'] = triplet_df_def[topic_col]
                    topic_col_found = topic_col
                    break

        triplet_df_kge['old_index'] = triplet_df_def.index

        # Configure column names
        if include_topics and topic_col_found:
            triplet_df_kge.columns = ['subject', 'relation', 'object',
                                      'new_topic', 'old_index']
        else:
            triplet_df_kge.columns = ['subject', 'relation', 'object', 'old_index']

        # Save CSV
        csv_path = os.path.join(output_dir, csv_filename)
        triplet_df_kge.to_csv(csv_path, index=False)

        print(f"Transformation completed!")
        print(f"Total triplets generated: {len(triplet_df_kge)}")
        print(f"CSV file saved to: {csv_path}")

    except Exception as e:
        print(f"ERROR during transformation: {e}")
        traceback.print_exc()


def explode_triplets(output_dir, output_name):
    """Generate triplets file from dataframe - SIN CAMBIOS"""
    output_path = os.path.join(output_dir, output_name)
    df = joblib.load(output_path)
    create_triplets_csv(df, output_dir, output_name)


# ============================================================================
# FUNCIONES PRINCIPALES MODIFICADAS (solo inicialización)
# ============================================================================

def generate_triplets_only(input_data, output_dir, output_name,
                           n_partitions, enable_pos_filtering, debug):
    """
    Mode 1: Generate triplets only without topic information
    MODIFICADO: Usa spaCy en lugar de CoreNLP
    """
    print("\n" + "=" * 50)
    print("MODE 1: TRIPLET GENERATION ONLY (spaCy)")
    print("=" * 50)

    # Verify input files
    if not os.path.exists(input_data):
        print(f"ERROR: Input data file not found: {input_data}")
        return

    # ========== CAMBIO: Inicializar spaCy en lugar de CoreNLP ==========
    print("\nInitializing spaCy...")
    try:
        nlp = spacy.load('en_core_web_sm')
        triplet_generator = TripletGeneratorSpacy()
        validator = ValidadorTripletasSpacy()
        print(f"✓ spaCy initialized")
    except Exception as e:
        print(f"ERROR: Could not load spaCy model")
        print(f"Error: {e}")
        print("\nInstall with: python -m spacy download en_core_web_sm")
        return
    # ====================================================================

    # Load data - IGUAL
    print(f"\nLoading data from: {input_data}")
    try:
        df = joblib.load(input_data)
        print(f"Data loaded: {len(df)} rows")
        ddf = dd.from_pandas(df, npartitions=n_partitions)
    except Exception as e:
        print(f"ERROR loading data: {e}")
        return

    # Main processing - IGUAL excepto parámetros
    print(f"\nStarting triplet generation with {n_partitions} partitions...")

    try:
        ddf['triplets'] = ddf.map_partitions(
            transform_partition_spacy,
            nlp=nlp,
            triplet_generator=triplet_generator,
            validator=validator,
            enable_pos_filtering=enable_pos_filtering,
            debug_rejected=debug,
            meta=('object')
        ).compute()
        print("Triplet generation completed!")
    except Exception as e:
        print(f"ERROR during processing: {e}")
        traceback.print_exc()
        return

    # Save results - IGUAL
    output_path = os.path.join(output_dir, output_name + "_only_triplets")
    print(f"\nSaving results to: {output_path}")

    try:
        df_result = ddf.compute()
        joblib.dump(df_result, output_path)
        print(f"Data saved correctly to: {output_path}")
    except Exception as e:
        print(f"ERROR saving data: {e}")
        return

    # Create CSV - IGUAL
    create_triplets_csv(
        df_result, output_dir,
        "dataset_triplet_amazon_only_triplets.csv",
        include_topics=False
    )

    print(f"\nGenerated files:")
    print(f"- Data with triplets: {output_path}")
    print(f"- Triplets CSV: {os.path.join(output_dir, 'dataset_triplet_amazon_only_triplets.csv')}")


def generate_triplets_with_topics(input_data, output_dir, output_name,
                                  n_partitions):
    """
    Mode 2: Generate triplets with topic information
    MODIFICADO: Usa spaCy en lugar de CoreNLP
    """
    print("\n" + "=" * 50)
    print("MODE 2: TRIPLET GENERATION WITH TOPICS (spaCy)")
    print("=" * 50)

    # Verify input files
    if not os.path.exists(input_data):
        print(f"ERROR: Input data file not found: {input_data}")
        return

    # ========== CAMBIO: Inicializar spaCy ==========
    print("\nInitializing spaCy...")
    try:
        nlp = spacy.load('en_core_web_sm')
        triplet_generator = TripletGeneratorSpacy()
        validator = ValidadorTripletasSpacy()
        print(f"✓ spaCy initialized")
    except Exception as e:
        print(f"ERROR: Could not load spaCy model")
        print(f"Error: {e}")
        return
    # ===============================================

    # Load data - IGUAL
    print(f"\nLoading data from: {input_data}")
    try:
        df = joblib.load(input_data)
        print(f"Data loaded: {len(df)} rows")
        print(f"Columns: {list(df.columns)}")
        ddf = dd.from_pandas(df, npartitions=n_partitions)
    except Exception as e:
        print(f"ERROR loading data: {e}")
        return

    # Main processing - IGUAL excepto parámetros
    print(f"\nStarting triplet generation with topics using {n_partitions} partitions...")

    try:
        ddf['triplets'] = ddf.map_partitions(
            transform_partition_spacy,
            nlp=nlp,
            triplet_generator=triplet_generator,
            validator=validator,
            enable_pos_filtering=True,
            debug_rejected=True,
            meta=('object')
        ).compute()
        print("Triplet generation with topics completed!")
    except Exception as e:
        print(f"ERROR during processing: {e}")
        traceback.print_exc()
        return

    # Save results - IGUAL
    output_path = os.path.join(output_dir, output_name)
    print(f"\nSaving results to: {output_path}")

    try:
        df_result = ddf.compute()
        joblib.dump(df_result, output_path)
        print(f"Data saved correctly to: {output_path}")
    except Exception as e:
        print(f"ERROR saving data: {e}")
        return

    print(df_result.shape)

    # Create CSV - IGUAL
    create_triplets_csv(
        df_result, output_dir,
        "dataset_triplet_amazon_new_simplified.csv",
        include_topics=True
    )

    print(f"\nGenerated files:")
    print(f"- Data with triplets: {output_path}")
    print(f"- Triplets CSV: {os.path.join(output_dir, 'dataset_triplet_amazon_new_simplified.csv')}")


def add_topics_to_existing_triplets(input_data, output_dir, output_name, existing_triplets):
    """
    Mode 3: Add topic information to existing triplets
    SIN CAMBIOS - No usa CoreNLP ni spaCy
    """
    print("\n" + "=" * 50)
    print("MODE 3: ADD TOPICS TO EXISTING TRIPLETS")
    print("=" * 50)

    if not os.path.exists(input_data):
        print(f"ERROR: Input data file not found: {input_data}")
        return

    if not existing_triplets or not os.path.exists(existing_triplets):
        print(f"ERROR: Existing triplets file not found: {existing_triplets}")
        return

    # Load data
    print(f"\nLoading updated data from: {input_data}")
    try:
        df_updated = joblib.load(input_data)
        print(f"Updated data loaded: {len(df_updated)} rows")
    except Exception as e:
        print(f"ERROR loading updated data: {e}")
        return

    # Load existing triplets
    print(f"\nLoading existing triplets from: {existing_triplets}")
    try:
        df_existing_triplets = pd.read_csv(existing_triplets)
        print(f"Existing triplets loaded: {len(df_existing_triplets)} rows")
    except Exception as e:
        print(f"ERROR loading existing triplets: {e}")
        return

    # Verify structure
    if 'triplets' not in df_existing_triplets.columns:
        print("ERROR: Existing triplets file does not contain 'triplets' column")
        return

    # Create mapping
    print("\nCreating topics mapping...")
    if 'topic' in df_updated.columns:
        topic_mapping = dict(zip(df_updated.index, df_updated['topic']))
    else:
        print("ERROR: Updated data does not contain 'topic' column")
        return

    # Update topics
    print("Updating topic information...")
    df_result = df_existing_triplets.copy()

    if hasattr(df_result, 'index'):
        df_result['topic'] = df_result.index.map(topic_mapping)
        print("Topics updated based on indices")

    # Save results
    output_path = os.path.join(output_dir, output_name + "_updated_topics")
    print(f"\nSaving updated results to: {output_path}")

    try:
        joblib.dump(df_result, output_path)
        print(f"Data with updated topics saved correctly")
    except Exception as e:
        print(f"ERROR saving data: {e}")
        return

    # Create CSV
    create_triplets_csv(
        df_result, output_dir,
        "dataset_triplet_amazon_updated_topics.csv",
        include_topics=True
    )

    print(f"\nGenerated files:")
    print(f"- Data with updated topics: {output_path}")
    print(f"- Updated triplets CSV: {os.path.join(output_dir, 'dataset_triplet_amazon_updated_topics.csv')}")


# ============================================================================
# MAIN - MODIFICADO: Sin CoreNLP endpoint
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Triplet Generation with spaCy')
    parser.add_argument('--input_data', type=str,
                        default='data/lda_eval/amazon/df_topic.pkl',
                        help='Path to input data file')
    parser.add_argument('--output_dir', type=str,
                        default='data/triples_raw/amazon',
                        help='Output directory for generated files')
    parser.add_argument('--output_name', type=str,
                        default='amazon_semantic_triplets_simplified-withtopic',
                        help='Base name for output files')
    # ========== ELIMINADO: corenlp_endpoint (ya no se usa) ==========
    parser.add_argument('--npartitions', type=int,
                        default=10,
                        help='Number of Dask partitions')
    # ========== ELIMINADO: sleep_time (spaCy no necesita) ==========
    parser.add_argument('--mode', type=str,
                        choices=['triplets_only', 'triplets_with_topics',
                                 'add_topics_to_existing', 'explode_triplets'],
                        default='triplets_with_topics',
                        help='Operation mode')
    parser.add_argument('--existing_triplets', type=str,
                        default='data/triples_raw/amazon/dataset_triplet_amazon_new_simplified.csv',
                        help='Path to existing triplets (for add_topics_to_existing mode)')
    parser.add_argument('--debug_rejected', action='store_true',
                        default=False,
                        help='Show detailed explanations of why triplets are rejected')

    # Parse arguments
    try:
        args = parser.parse_args()
    except SystemExit:
        args = argparse.Namespace(
            input_data='data/lda_eval/amazon/df_topic.pkl',
            output_dir='data/triples_raw/amazon',
            output_name='amazon_semantic_triplets_simplified-withtopic',
            npartitions=10,
            mode='triplets_with_topics',
            existing_triplets=None,
            debug_rejected=False
        )

    # Configuration
    INPUT_DATA = args.input_data
    OUTPUT_DIR = args.output_dir
    OUTPUT_NAME = args.output_name
    N_PARTITIONS = args.npartitions
    MODE = args.mode
    EXISTING_TRIPLETS = args.existing_triplets
    DEBUG_REJECTED = args.debug_rejected

    # Display configuration
    print("=" * 60)
    print("TRIPLET GENERATION CONFIGURATION (spaCy)")
    print("=" * 60)
    print(f"Operation mode: {MODE}")
    print(f"Input file: {INPUT_DATA}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Output name: {OUTPUT_NAME}")

    if MODE == 'add_topics_to_existing':
        print(f"Existing triplets file: {EXISTING_TRIPLETS}")
    else:
        print(f"Dask partitions: {N_PARTITIONS}")
        print(f"Debug rejected: {'Yes' if DEBUG_REJECTED else 'No'}")
    print("=" * 60)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Execute mode
    if MODE == 'triplets_only':
        generate_triplets_only(INPUT_DATA, OUTPUT_DIR, OUTPUT_NAME,
                               N_PARTITIONS, True, DEBUG_REJECTED)
    elif MODE == 'triplets_with_topics':
        generate_triplets_with_topics(INPUT_DATA, OUTPUT_DIR, OUTPUT_NAME, N_PARTITIONS)
    elif MODE == 'add_topics_to_existing':
        add_topics_to_existing_triplets(INPUT_DATA, OUTPUT_DIR, OUTPUT_NAME, EXISTING_TRIPLETS)
    else:
        explode_triplets(OUTPUT_DIR, OUTPUT_NAME)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MULTI-MODE TRIPLET GENERATOR (spaCy Optimized)")
    print("=" * 60)
    print("Available modes:")
    print("1. triplets_only: Generate triplets only")
    print("2. triplets_with_topics: Generate triplets with topics")
    print("3. add_topics_to_existing: Add topics to existing triplets")
    print("4. explode_triplets: Generate triplets file from dataframe")
    print("=" * 60)

    main()