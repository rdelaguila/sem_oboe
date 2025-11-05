#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import pandas as pd
import stanza
from stanza.server import CoreNLPClient
import joblib
import dask.dataframe as dd
import traceback
import importlib
from time import sleep
from utils.triplet_manager_lib import TripletManager, TripletGenerator, ValidadorTripletas


# ============================================================================
# GLOBAL HELPER FUNCTIONS
# ============================================================================

def return_triplets(sentence, phrase, pos, triplet_generator):
    """Extract triplets from a sentence using different methods"""
    triplets = [triplet_generator.encapsulate((triple.subject, triple.relation, triple.object), True)
                 for triple in sentence.openieTriple]


    if len(triplets) == 0:
        triplet = triplet_generator.triplet_extraction(phrase)
        if triplet is None:
            return triplets

    if triplet_generator._detect_adj_noun(pos)[0] != -1:
        triplet = triplet_generator.generate_triplet_adj_noun(pos)
    if triplet is not None:
        triplet = triplet_generator.encapsulate(triplet, True)
        if type(triplet) == list:
            triplets.extend(triplet)
        else:
            triplets.append(triplet)

    if triplet_generator._detect_nn_nnp(pos)[0] != -1:
        triplet = triplet_generator.generate_triplet_nn_nnp(pos)
        if triplet is not None:
            triplet = triplet_generator.encapsulate(triplet, True)
            if type(triplet) == list:
                triplets.extend(triplet)
            else:
                triplets.append(triplet)

    if triplet_generator._detect_nn_of_nn(pos)[0] != -1:
        triplet = triplet_generator.generate_triplet_nn_of_nn(pos)
        if triplet is not None:
            triplet = triplet_generator.encapsulate(triplet, True)
            if type(triplet) == list:
                triplets.extend(triplet)
            else:
                triplets.append(triplet)

    if triplet_generator._detect_nn_place_nn(pos)[0] != -1:
        triplet = triplet_generator.generate_triplet_nn_place_nn(pos)
        if triplet is not None:
            triplet = triplet_generator.encapsulate(triplet, True)
            if type(triplet) == list:
                triplets.extend(triplet)
            else:
                triplets.append(triplet)

    triplet = triplet_generator.generate_triplet_adjectives_nn_adjs(pos)
    if triplet is not None:
        triplet = triplet_generator.encapsulate(triplet, True)
        if type(triplet) == list:
            triplets.extend(triplet)
        else:
            triplets.append(triplet)

    return triplets


def return_pos(sentence):
    """Extract POS information from a sentence"""
    dictionary = dict()
    for position in (sentence.ListFields()[0][1]):
        dictionary[position.word] = position.pos
    return dictionary


# Function to use in processing
def annotate_row_optimized(row, client, sleep_time, enable_pos_filtering=True, debug_rejected=True):
    """
    Optimized version with basic validation + POS
    """
    triplet_generator = TripletGenerator()
    validator = ValidadorTripletas()
    result = []

    # Get basic information
    topic = row.get('topic')

    # Process the text
    text_column = ''
    for text_col in ['text_coref', 'coref_text', 'text', 'news']:
        if text_col in row and row[text_col]:
            text_column = str(row[text_col])
            break

    if not text_column:
        return result

    for phrase in text_column.split('.'):
        if not phrase.strip():
            continue
        try:
            annotation = client.annotate(phrase)
            sleep(sleep_time)

            for sentence in annotation.sentence:
                d = return_pos(sentence)
                t = return_triplets(sentence, phrase, d, triplet_generator)

                if enable_pos_filtering:
                    # Apply validation with debug
                    valid_triplets = []
                    for triplet in t:
                        if debug_rejected:
                            is_valid, explanation = validator.quick_validation_with_pos(triplet, d, debug=True)
                            if is_valid:
                                valid_triplets.append(triplet)
                                print(
                                    f"✓ ACCEPTED: {triplet.get('subject')} | {triplet.get('relation')} | {triplet.get('object')}")
                                print(f"    Reason: {explanation}")
                            else:
                                print(
                                    f"✗ REJECTED: {triplet.get('subject')} | {triplet.get('relation')} | {triplet.get('object')}")
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


def transform_partition_optimized(df, client, sleep_time, enable_pos_filtering, debug_rejected):
    """
    Transforms a dataframe partition applying optimized annotation to each row
    """
    return df.apply(
        lambda row: annotate_row_optimized(row, client, sleep_time, enable_pos_filtering, debug_rejected),
        axis=1
    )

def annotate_row(row, client, sleep_time, enable_filtering, filter_method):
    """
    Process a dataframe row and extract triplets
    """
    triplet_generator = TripletGenerator()
    result = []

    # Debug: show available columns in first row
    #if not hasattr(annotate_row, '_debug_shown'):
    #    print(f"DEBUG: Available columns: {list(row.index)}")
    #    for col in ['topic', 'new_target', 'target', 'entidades', 'entidades_dbpedia',
    #                'entidades_dbpedia_simplificadas']:
    #        if col in row:
    #            print(f"DEBUG: {col} = {type(row[col])} - {str(row[col])[:100]}...")
    #    annotate_row._debug_shown = True

    # Get topic information - search for different column names
    topic = row.get('topic')

    # Get data for filtering if enabled
    triplet_manager = None
    ner = set()
    dbpedia = []

    if enable_filtering:
        triplet_manager = TripletManager()

        # Get NER entities - search for different column names
        if filter_method in ['ner_only', 'both']:
            for ner_col in ['entidades', 'ner', 'tokens']:
                if ner_col in row and row[ner_col]:
                    ner = row[ner_col]
                    if isinstance(ner, str):
                        ner = set([ner])
                    elif not isinstance(ner, set):
                        ner = set(ner) if ner else set()
                    break

        # Get DBpedia entities - search for different column names
        if filter_method in ['dbpedia_only', 'both']:
            for dbpedia_col in ['entidades_dbpedia_simplificadas', 'entidades_dbpedia']:
                if dbpedia_col in row and row[dbpedia_col]:
                    dbpedia_entities = row[dbpedia_col]

                    # Handle different data types
                    if isinstance(dbpedia_entities, dict):
                        dbpedia = list(dbpedia_entities.keys())
                        break
                    elif isinstance(dbpedia_entities, str):
                        # If string, try to evaluate it as dictionary
                        try:
                            import ast
                            dbpedia_entities = ast.literal_eval(dbpedia_entities)
                            if isinstance(dbpedia_entities, dict):
                                dbpedia = list(dbpedia_entities.keys())
                                break
                        except:
                            # If cannot be evaluated, treat as list of strings
                            try:
                                dbpedia = [dbpedia_entities] if dbpedia_entities else []
                                break
                            except:
                                continue
                    elif isinstance(dbpedia_entities, (list, tuple)):
                        dbpedia = list(dbpedia_entities)
                        break

    # Process the text - search for different column names
    text_column = ''
    for text_col in ['text_coref', 'coref_text', 'text', 'news']:
        if text_col in row and row[text_col]:
            text_column = str(row[text_col])
            break

    if not text_column:
        return result

    for phrase in text_column.split('.'):
        if not phrase.strip():
            continue
        try:
            annotation = client.annotate(phrase)
            sleep(sleep_time)
            #print(f"""annotation {annotation}""")

            for sentence in annotation.sentence:
                d = return_pos(sentence)

                t = return_triplets(sentence, phrase, d, triplet_generator)
                # Apply filtering if enabled
                if enable_filtering and triplet_manager:
                    filtered_triplets = []
                    for triplet in t:
                        try:
                            if triplet_manager.is_candidate(triplet, d, ner, topic, dbpedia):
                                filtered_triplets.append(triplet)
                        except:
                            filtered_triplets.append(triplet)
                    result.extend(filtered_triplets)
                else:
                    result.extend(t)
        except:
            continue

    return result


def transform_partition(df, client, sleep_time, enable_filtering, filter_method):
    """
    Transforms a dataframe partition applying annotation to each row
    """
    return df.apply(
        lambda row: annotate_row(row, client, sleep_time, enable_filtering, filter_method),
        axis=1
    )


# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def main():
    # ---------- COMMAND-LINE ARGUMENTS ----------
    parser = argparse.ArgumentParser(description='Triplet Generation from Dataset')
    parser.add_argument('--input_data', type=str,
                        default='../data/lda_eval/arxiv/df_topic.pkl',
                        help='Path to input data file')
    parser.add_argument('--output_dir', type=str,
                        default='../data/triples_raw/arxiv',
                        help='Output directory for generated files')
    parser.add_argument('--output_name', type=str,
                        default='arxiv_semantic_triplets_simplified-withtopic',
                        help='Base name for output files')
    parser.add_argument('--corenlp_endpoint', type=str,
                        default='http://0.0.0.0:9000',
                        help='CoreNLP server endpoint')
    parser.add_argument('--npartitions', type=int,
                        default=10,
                        help='Number of Dask partitions')
    parser.add_argument('--sleep_time', type=float,
                        default=0.30,
                        help='Sleep time between annotations (seconds)')
    parser.add_argument('--mode', type=str,
                        choices=['triplets_only', 'triplets_with_topics', 'add_topics_to_existing', 'explode_triplets'],
                        default='triplets_with_topics',
                        help='Operation mode: 1=triplets only, 2=triplets+topics, 3=add topics to existing, 4=generate triplets file')
    parser.add_argument('--existing_triplets', type=str,
                        default='data/triples_raw/amazon/dataset_triplet_amazon_new_simplified.csv',

                        help='Path to existing triplets (for add_topics_to_existing mode)')
    parser.add_argument('--enable_filtering', action='store_true',
                        default=False,
                        help='Enable vocabulary filtering using DBpedia and NER entities (default: False)')
    parser.add_argument('--filter_method', type=str,
                        choices=['dbpedia_only', 'ner_only', 'both'],
                        default='both',
                        help='Filtering method when enabled: dbpedia_only, ner_only, or both (default: both)')

    parser.add_argument('--debug_rejected', action='store_true',
                        default=False,
                        help='Show detailed explanations of why triplets are rejected')

    # Parse arguments with error handling
    try:
        args = parser.parse_args()
    except SystemExit:
        # If no valid arguments provided, use defaults
        print("No valid arguments provided. Using default values:")
        args = argparse.Namespace(
            input_data='data/lda_eval/amazon/df_topic.pkl',
            output_dir='../olds/data/triples_raw/amazon',
            output_name='amazon_semantic_triplets_simplified-withtopic',
            corenlp_endpoint='http://0.0.0.0:9000',
            npartitions=10,
            sleep_time=0.30,
            mode='triplets_with_topics',
            existing_triplets=None,
            enable_filtering=False,
            filter_method='both'
        )

    # ---------- CONFIGURATION ----------
    INPUT_DATA = args.input_data
    OUTPUT_DIR = args.output_dir
    OUTPUT_NAME = args.output_name
    CORENLP_ENDPOINT = args.corenlp_endpoint
    N_PARTITIONS = args.npartitions
    SLEEP_TIME = args.sleep_time
    MODE = args.mode
    EXISTING_TRIPLETS = args.existing_triplets
    print(EXISTING_TRIPLETS)
    ENABLE_FILTERING = args.enable_filtering
    FILTER_METHOD = args.filter_method

    # Display configuration
    print("=" * 60)
    print("TRIPLET GENERATION CONFIGURATION")
    print("=" * 60)
    print(f"Operation mode: {MODE}")
    print(f"Input file: {INPUT_DATA}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Output name: {OUTPUT_NAME}")

    if MODE == 'triplets_with_topics':
        print(f"Existing triplets file: {EXISTING_TRIPLETS}")
    else:
        print(f"CoreNLP endpoint: {CORENLP_ENDPOINT}")
        print(f"Dask partitions: {N_PARTITIONS}")
        print(f"Sleep time: {SLEEP_TIME}s")
        print(f"Filtering enabled: {'Yes' if ENABLE_FILTERING else 'No'}")
        if ENABLE_FILTERING:
            print(f"Filtering method: {FILTER_METHOD}")
    print("=" * 60)

    # Create output directory if not exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Execute according to selected mode
    if MODE == 'triplets_only':
        generate_triplets_only(INPUT_DATA, OUTPUT_DIR, OUTPUT_NAME, CORENLP_ENDPOINT,
                               N_PARTITIONS, SLEEP_TIME, ENABLE_FILTERING, FILTER_METHOD)
    elif MODE == 'triplets_with_topics':
        generate_triplets_with_topics(INPUT_DATA, OUTPUT_DIR, OUTPUT_NAME,
                                      CORENLP_ENDPOINT, N_PARTITIONS, SLEEP_TIME, ENABLE_FILTERING, FILTER_METHOD)
    elif MODE == 'add_topics_to_existing':
        print(EXISTING_TRIPLETS)
        add_topics_to_existing_triplets(INPUT_DATA, OUTPUT_DIR, OUTPUT_NAME, EXISTING_TRIPLETS)
    else:

        explode_triplets(OUTPUT_DIR, OUTPUT_NAME,)

def generate_triplets_only(input_data, output_dir, output_name, corenlp_endpoint,
                           n_partitions, sleep_time, enable_pos_filtering, debug):
    """
    Mode 1: Generate triplets only without topic information
    """
    print("\n" + "=" * 50)
    print("MODE 1: TRIPLET GENERATION ONLY")
    print("=" * 50)

    # Verify input files
    if not os.path.exists(input_data):
        print(f"ERROR: Input data file not found: {input_data}")
        return

    # Initialize CoreNLP
    print("\nInitializing CoreNLP client...")
    try:
        client = CoreNLPClient(
            annotators=['openie'],
            endpoint=corenlp_endpoint,
            start_server=True,
            be_quiet=True
        )
        print(f"CoreNLP client initialized")
    except Exception as e:
        print(f"ERROR: Could not connect to CoreNLP at {corenlp_endpoint}")
        print(f"Error: {e}")
        return

    # Load data
    print(f"\nLoading data from: {input_data}")
    try:
        df = joblib.load(input_data)
        print(f"Data loaded: {len(df)} rows")
        ddf = dd.from_pandas(df, npartitions=n_partitions)
    except Exception as e:
        print(f"ERROR loading data: {e}")
        return

    # Main processing
    print(f"\nStarting triplet generation with {n_partitions} partitions...")

    try:
        # Use map_partitions with necessary parameters
        ddf['triplets'] = ddf.map_partitions(
            transform_partition_optimized,
            client=client,
            sleep_time=sleep_time,
            enable_pos_filtering=enable_pos_filtering,
            debug_rejected=debug,
            meta=('object')
        ).compute()
        print("Triplet generation completed!")
    except Exception as e:
        print(f"ERROR during processing: {e}")
        traceback.print_exc()
        return

    # Save results
    output_path = os.path.join(output_dir, output_name + "_only_triplets")
    print(f"\nSaving results to: {output_path}")

    try:
        df_result = ddf.compute()
        joblib.dump(df_result, output_path)
        print(f"Data saved correctly to: {output_path}")
    except Exception as e:
        print(f"ERROR saving data: {e}")
        return

    # Create CSV of triplets without topics
    create_triplets_csv(df_result, output_dir, "dataset_triplet_amazon_only_triplets.csv", include_topics=False)

    print(f"\nGenerated files:")
    print(f"- Data with triplets: {output_path}")
    print(f"- Triplets CSV: {os.path.join(output_dir, 'dataset_triplet_amazon_only_triplets.csv')}")


def generate_triplets_with_topics(input_data, output_dir, output_name,
                                  corenlp_endpoint, n_partitions, sleep_time, enable_filtering, filter_method):
    """
    Mode 2: Generate triplets with topic information (original behavior)
    """
    print("\n" + "=" * 50)
    print("MODE 2: TRIPLET GENERATION WITH TOPICS")
    print("=" * 50)

    # Verify input files
    if not os.path.exists(input_data):
        print(f"ERROR: Input data file not found: {input_data}")
        return

    # Initialize CoreNLP
    print("\nInitializing CoreNLP client...")
    try:
        client = CoreNLPClient(
            annotators=['openie'],
            endpoint=corenlp_endpoint,
            start_server=False,
            be_quiet=True
        )
        print(f"CoreNLP client initialized")
    except Exception as e:
        print(f"ERROR: Could not connect to CoreNLP at {corenlp_endpoint}")
        print(f"Error: {e}")
        return

    # Load data and topics
    print(f"\nLoading data from: {input_data}")
    try:
        df = joblib.load(input_data)
        print(f"Data loaded: {len(df)} rows")
        print(f"Columns are {list(df.columns)}")
        ddf = dd.from_pandas(df, npartitions=n_partitions)
    except Exception as e:
        print(f"ERROR loading data: {e}")
        return

    # Main processing
    print(f"\nStarting triplet generation with topics using {n_partitions} partitions...")

    try:
        # Use map_partitions with necessary parameters
        ddf['triplets'] = ddf.map_partitions(
            transform_partition_optimized,
            client=client,
           #topics=topics,
            sleep_time=sleep_time,
            debug_rejected=True,  # <-- New parameter

            enable_pos_filtering=True,
            meta=('object')
        ).compute()
        print("Triplet generation with topics completed!")
    except Exception as e:
        print(f"ERROR during processing: {e}")
        traceback.print_exc()
        return

    # Save results
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
    # Create CSV of triplets with topics
    create_triplets_csv(df_result, output_dir, "dataset_triplet_amazon_new_simplified.csv", include_topics=True)

    print(f"\nGenerated files:")
    print(f"- Data with triplets: {output_path}")
    print(f"- Triplets CSV: {os.path.join(output_dir, 'dataset_triplet_amazon_new_simplified.csv')}")


def add_topics_to_existing_triplets(input_data, topics_file, output_dir, output_name, existing_triplets):
    """
    Mode 3: Add topic information to existing triplets
    """
    print("\n" + "=" * 50)
    print("MODE 3: ADD TOPICS TO EXISTING TRIPLETS")
    print("=" * 50)

    # Verify input files
    if not os.path.exists(input_data):
        print(f"ERROR: Input data file not found: {input_data}")
        return

    if not os.path.exists(topics_file):
        print(f"ERROR: Topics file not found: {topics_file}")
        return

    if not existing_triplets or not os.path.exists(existing_triplets):
        print(f"ERROR: Existing triplets file not found: {existing_triplets}")
        return

    # Load updated data with new topic information
    print(f"\nLoading updated data from: {input_data}")
    try:
        df_updated = joblib.load(input_data)
        print(f"Updated data loaded: {len(df_updated)} rows")
    except Exception as e:
        print(f"ERROR loading updated data: {e}")
        return

    # Load new topics
    print(f"\nLoading new topics from: {topics_file}")
    try:
        topics = joblib.load(topics_file)
        print(f"New topics loaded successfully")
    except Exception as e:
        print(f"ERROR loading topics: {e}")
        return

    # Load existing triplets
    print(f"\nLoading existing triplets from: {existing_triplets}")
    try:
        df_existing_triplets = pd.read_csv(existing_triplets)
        print(f"Existing triplets loaded: {len(df_existing_triplets)} rows")
    except Exception as e:
        print(f"ERROR loading existing triplets: {e}")
        return

    # Verify existing triplets have necessary column
    if 'triplets' not in df_existing_triplets.columns:
        print("ERROR: Existing triplets file does not contain 'triplets' column")
        return

    # Create mapping of indices to new topics
    print("\nCreating topics mapping...")
    if 'topic' in df_updated.columns:
        topic_mapping = dict(zip(df_updated.index, df_updated['topic']))
    else:
        print("ERROR: Updated data does not contain 'topic' column")
        return

    # Update topic information in existing triplets
    print("Updating topic information...")
    df_result = df_existing_triplets.copy()

    # Update topics column if exists, or create new one
    if hasattr(df_result, 'index'):
        df_result['topic'] = df_result.index.map(topic_mapping)
        print("Topics updated based on indices")
    else:
        print("WARNING: Could not map topics automatically")
        print("Original topics will be kept if they exist")

    # Copy other relevant columns from updated data if necessary
    common_columns = set(df_updated.columns) & set(df_result.columns)
    for col in common_columns:
        if col not in ['triplets', 'topic']:  # Do not overwrite these critical columns
            try:
                df_result[col] = df_updated[col]
                print(f"Column '{col}' updated")
            except:
                print(f"Could not update column '{col}'")

    # Save results
    output_path = os.path.join(output_dir, output_name + "_updated_topics")
    print(f"\nSaving updated results to: {output_path}")

    try:
        joblib.dump(df_result, output_path)
        print(f"Data with updated topics saved correctly")
    except Exception as e:
        print(f"ERROR saving data: {e}")
        return

    # Create new CSV of triplets with updated topics
    create_triplets_csv(df_result, output_dir, "dataset_triplet_amazon_updated_topics.csv", include_topics=True)

    print(f"\nGenerated files:")
    print(f"- Data with updated topics: {output_path}")
    print(f"- Updated triplets CSV: {os.path.join(output_dir, 'dataset_triplet_amazon_updated_topics.csv')}")


def deserialize_triplets(triplet_str):
    """Convert triplet string to actual list"""
    if pd.isna(triplet_str) or triplet_str == '' or triplet_str == '[]':
        return []

    try:
        # Try eval (use with trusted data only)
        import ast
        return ast.literal_eval(triplet_str)
    except:
        try:
            # Alternative with eval if ast fails
            return eval(triplet_str)
        except:
            print(f"WARNING: Could not deserialize: {triplet_str[:100]}...")
            return []


def create_triplets_csv(df, output_dir, csv_filename, include_topics=True):
    """
    Helper function to create triplets CSV
    """
    print(f"\nTransforming to triplets dataframe...")

    try:
        # Remove rows without triplets
        original_len = len(df)
        df_filtered = df[df['triplets'].map(len) > 0].copy()
        filtered_len = len(df_filtered)
        print(f"Rows filtered: {original_len} -> {filtered_len} ({original_len - filtered_len} without triplets)")

        if filtered_len == 0:
            print("WARNING: No triplets to process")
            return

        # Explode triplets

        #triplet_df_def = df_filtered.explode('triplets')

        #triplet_df_def = triplet_df_def.reset_index()

        df_filtered['triplets'] = df_filtered['triplets'].apply(deserialize_triplets)

        triplet_df_def = df_filtered.explode('triplets', ignore_index=True)

        # Filter possible None values from explosion
        triplet_df_def = triplet_df_def[triplet_df_def['triplets'].notna()]
        triplet_df_kge = pd.DataFrame(triplet_df_def['triplets'].tolist(), index=triplet_df_def.index)

        # Add metadata - search for different topic column names
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
            triplet_df_kge.columns = ['subject', 'relation', 'object', 'new_topic', 'old_index']
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
    """Function that will generate the triplets and topics file from a dataframe containing both columns"""

    output_path = os.path.join(output_dir, output_name)

    df = joblib.load(output_path)
    create_triplets_csv(df,output_dir,output_name)

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MULTI-MODE TRIPLET GENERATOR")
    print("=" * 60)
    print("Available modes:")
    print("1. triplets_only: Generate triplets only")
    print("2. triplets_with_topics: Generate triplets with topics")
    print("3. add_topics_to_existing: Add topics to existing triplets")
    print("")
    print("Filtering options:")
    print("--enable_filtering: Enable entity-based filtering (default: disabled)")
    print("--filter_method: Filtering method (dbpedia_only, ner_only, both)")
    print("=" * 60)

    main()