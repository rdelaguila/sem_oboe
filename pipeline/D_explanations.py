# File: topic_explanation_algorithm_enhanced.py
"""
Implementation with advanced prompting strategies based on QualIT and XAI surveys:
- Key phrase extraction
- Hallucination verification
- Hierarchical clustering
- Chain-of-thought explanation generation
- Evaluation with specific criteria
- Dynamic configuration of datasets and vocabularies
- Support for analyzing single topic or all topics
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
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.manifold import TSNE

# Try to import UMAP, but don't fail if not installed
try:
    from umap import UMAP

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("⚠️ UMAP not installed. Install with: pip install umap-learn")
    print("   UMAP visualizations will be skipped for spectral clustering.")

import spacy
import torch
from utils.types import StringCaseInsensitiveSet, CaseInsensitiveDict, CaseInsensitiveSet
from utils.triplet_manager_lib import Tripleta
from operator import itemgetter
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, T5Tokenizer, T5ForConditionalGeneration
import joblib
import nltk
from nltk.corpus import wordnet as wn
import pickle
from typing import Dict, List, Set, Union, Any


# ======== DYNAMIC CONFIGURATION ========
def load_processed_dataframe(repo_name: str) -> pd.DataFrame:
    """
    Loads the processed dataframe for the specified repository
    """
    processed_path = f'../data/processed/{repo_name}/{repo_name}_processed_semantic.pkl'

    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"File not found: {processed_path}")

    with open(processed_path, 'rb') as f:
        data = joblib.load(f)
        print(data.head())

    if isinstance(data, pd.DataFrame):
        print(f"Data loaded as DataFrame: {len(data)} rows")
        df = data
    else:
        raise TypeError(f"Unsupported data type: {type(data)}. Expected DataFrame")

    return df


def create_vocabulary_dictionaries(df: pd.DataFrame) -> Dict[int, Dict[str, List[str]]]:
    """
    Creates NER and DBPedia vocabulary dictionaries by topic
    Adapted for the new amazon data format
    """
    vocabulary_dict = {}

    print(f"DataFrame structure:")
    print(f"  Columns: {list(df.columns) if hasattr(df, 'columns') else 'Not available'}")
    print(f"  Shape: {df.shape if hasattr(df, 'shape') else 'Not available'}")

    # Show data sample for debugging
    print(f"Data sample:")
    if hasattr(df, 'head'):
        print(df.head())
    else:
        print("  DataFrame without head method available")

    # Automatically identify relevant columns
    possible_ner_columns = []
    possible_dbpedia_columns = []
    topic_name = ''

    if hasattr(df, 'columns'):
        print(f"Searching for relevant columns...")
        for col_idx, col in enumerate(df.columns):
            col_lower = str(col).lower()
            print(f"  Column {col_idx}: {col} (type: {type(df.iloc[0, col_idx]) if len(df) > 0 else 'N/A'})")

            if 'ner' in col_lower or 'entidad' in col_lower or 'entidades' in col_lower or 'entity' in col_lower:
                possible_ner_columns.append(col)
                print(f"    -> Detected as NER")
            if 'dbpedia' in col_lower or 'entidades_dbpedia' in col_lower or 'dbp' in col_lower or 'uri' in col_lower:
                possible_dbpedia_columns.append(col)
                print(f"    -> Detected as DBPedia")
            if 'target' in col_lower or 'topic' in col_lower or 'topic_new' in col_lower or 'new_target' in col_lower or 'new_topic' in col_lower:
                topic_name = col
                print(f"    -> Detected as topic column")

    print(f"Detected NER columns: {possible_ner_columns}")
    print(f"Detected DBPedia columns: {possible_dbpedia_columns}")
    if topic_name == '':
        raise Exception('no topic detected')
    print(f"Detected topic column: {topic_name}")

    # If no specific columns detected, search automatically by content
    if not possible_ner_columns and not possible_dbpedia_columns:
        print("No specific columns detected. Searching by content...")

        # Examine all columns to identify structure
        for col_idx, col_name in enumerate(df.columns):
            if len(df) > 0:
                sample_value = df.iloc[0, col_idx]
                print(f"  Analyzing column {col_idx} ({col_name}):")
                print(f"      Type: {type(sample_value)}")

                # If it's a set, probably NER
                if isinstance(sample_value, set):
                    possible_ner_columns.append(col_name)
                    print(f"      -> Identified as NER (set with {len(sample_value)} elements)")
                    if len(sample_value) > 0:
                        print(f"      -> Sample: {list(sample_value)[:3]}")

                # If it's a dictionary, check if it has DBPedia structure
                elif isinstance(sample_value, dict):
                    print(f"      -> Dictionary with {len(sample_value)} keys")
                    if len(sample_value) > 0:
                        first_key = list(sample_value.keys())[0]
                        first_value = sample_value[first_key]
                        print(f"      -> First key: {first_key}")
                        print(f"      -> First value type: {type(first_value)}")

                        # Verify new DBPedia structure
                        if isinstance(first_value, dict):
                            print(f"      -> Value keys: {list(first_value.keys())}")
                            if 'URI' in first_value:
                                possible_dbpedia_columns.append(col_name)
                                print(f"      -> ✅ Identified as DBPedia (dict with URI)")
                            elif 'uri' in str(first_value).lower():
                                possible_dbpedia_columns.append(col_name)
                                print(f"      -> ✅ Identified as DBPedia (contains uri)")
                else:
                    print(f"      -> Type {type(sample_value)} not recognized as NER/DBPedia")

    print(f"Final NER columns: {possible_ner_columns}")
    print(f"Final DBPedia columns: {possible_dbpedia_columns}")

    # Process each DataFrame row
    for idx, row in df.iterrows():
        # Determine topic_id
        if topic_name and topic_name in row:
            topic_id = row[topic_name]
        else:
            topic_id = idx  # Use index if no topic column

        print(f"\nProcessing topic {topic_id} (row {idx}):")

        # Process NER entities
        ner_entities = []
        for ner_col in possible_ner_columns:
            if ner_col in row and row[ner_col] is not None:
                data = row[ner_col]
                print(f"    Processing NER from column '{ner_col}' (type: {type(data)})")

                if isinstance(data, set):
                    entities = [str(x).strip().lower() for x in data if x and str(x).strip()]
                    ner_entities.extend(entities)
                    print(f"        ✅ Extracted {len(entities)} NER entities")
                    if entities:
                        print(f"        Examples: {entities[:3]}")

                elif isinstance(data, list):
                    entities = [str(x).strip().lower() for x in data if x and str(x).strip()]
                    ner_entities.extend(entities)
                    print(f"        ✅ Extracted {len(entities)} NER entities from list")

                elif isinstance(data, str):
                    entities = [x.strip().lower() for x in data.replace(',', ' ').split() if x.strip()]
                    ner_entities.extend(entities)
                    print(f"        ✅ Extracted {len(entities)} NER entities from string")

                elif isinstance(data, dict):
                    entities = [str(k).strip().lower() for k in data.keys() if k]
                    ner_entities.extend(entities)
                    print(f"        ✅ Extracted {len(entities)} NER entities from dict keys")

        # Process DBPedia entities (new format)
        dbpedia_entities = []
        for dbp_col in possible_dbpedia_columns:
            if dbp_col in row and row[dbp_col] is not None:
                data = row[dbp_col]
                print(f"    Processing DBPedia from column '{dbp_col}' (type: {type(data)})")
                print(f"        Dictionary with {len(data)} entities")

                if isinstance(data, dict):
                    # New format: {'dog': {'URI': '...', 'types': ...}, ...}
                    for entity_name, entity_data in data.items():
                        if isinstance(entity_data, dict):
                            # Add entity name
                            entity_clean = str(entity_name).strip().lower()
                            if entity_clean and len(entity_clean) > 1:
                                dbpedia_entities.append(entity_clean)

                            # Extract types if available
                            types_data = entity_data.get('types', '')
                            if isinstance(types_data, str) and types_data:
                                print(f"        Types for '{entity_name}': {types_data}")
                                # Extract type names (e.g. 'DBpedia:Food' -> 'food')
                                for type_part in types_data.split(','):
                                    type_part = type_part.strip()
                                    if ':' in type_part:
                                        type_name = type_part.split(':')[-1].strip().lower()
                                        if type_name and not type_name.startswith('q') and len(type_name) > 1:
                                            dbpedia_entities.append(type_name)

                            # Also extract from normalized_types if exists
                            types_norm = entity_data.get('types_normalizados', '')
                            if isinstance(types_norm, str) and types_norm:
                                print(f"        Normalized types for '{entity_name}': {types_norm}")
                                for type_part in types_norm.split():
                                    type_part = type_part.strip().lower()
                                    if type_part and not type_part.startswith('q') and len(type_part) > 1:
                                        dbpedia_entities.append(type_part)

                    print(f"        ✅ Extracted {len(set(dbpedia_entities))} unique DBPedia entities")
                    if dbpedia_entities:
                        print(f"        Examples: {list(set(dbpedia_entities))[:5]}")

        # If no specific data found, search in all columns
        if not ner_entities and not dbpedia_entities:
            print(f"    No specific data found for topic {topic_id}. Searching in all columns.")
            for col_name, value in row.items():
                if value is not None:
                    if isinstance(value, set):
                        # Treat sets as NER
                        entities = [str(x).strip().lower() for x in value if x and str(x).strip()]
                        ner_entities.extend(entities)
                        print(f"        Found {len(entities)} NER entities in column '{col_name}'")

                    elif isinstance(value, dict):
                        # Search for DBPedia structure
                        for k, v in value.items():
                            if isinstance(v, dict) and ('URI' in v or 'types' in v or 'uri' in str(v).lower()):
                                entity_name = str(k).strip().lower()
                                if entity_name and len(entity_name) > 1:
                                    dbpedia_entities.append(entity_name)

                                # Extract types
                                types_data = v.get('types', '')
                                if isinstance(types_data, str) and types_data:
                                    for type_part in types_data.split(','):
                                        if ':' in type_part:
                                            type_name = type_part.split(':')[-1].strip().lower()
                                            if type_name and len(type_name) > 1:
                                                dbpedia_entities.append(type_name)

                        if dbpedia_entities:
                            print(f"        Found DBPedia entities in column '{col_name}'")

        # Clean and remove duplicates
        ner_entities = list(set([x for x in ner_entities if x and len(x) > 1]))
        dbpedia_entities = list(set([x for x in dbpedia_entities if x and len(x) > 1]))

        vocabulary_dict[topic_id] = {
            'ner': ner_entities,
            'dbpedia': dbpedia_entities
        }

        print(f"    ✅ Topic {topic_id}: {len(ner_entities)} NER, {len(dbpedia_entities)} DBPedia")
        if dbpedia_entities:
            print(f"        DBPedia examples: {dbpedia_entities[:5]}")
        if ner_entities:
            print(f"        NER examples: {ner_entities[:5]}")

    return vocabulary_dict


def load_top_terms_by_topic(repo_name: str) -> Dict[int, List[str]]:
    """
    Loads the most relevant terms by topic
    """
    top_terms_path = f'../data/lda_eval/{repo_name}/top_terms_by_topic.pkl'

    if not os.path.exists(top_terms_path):
        raise FileNotFoundError(f"File not found: {top_terms_path}")

    with open(top_terms_path, 'rb') as f:
        top_terms = joblib.load(f)  # Changed from pickle.load to joblib.load for consistency

    return top_terms


def show_topic_vocabulary(topic_id: int, vocabulary_dict: Dict[int, Dict[str, List[str]]],
                          top_terms: Dict[int, List[str]]) -> None:
    """
    Shows relevant vocabulary for a specific topic
    """
    print(f"\n=== VOCABULARY FOR TOPIC {topic_id} ===")

    # Show most relevant terms from LDA
    if topic_id in top_terms:
        print(f"\nMost relevant terms (LDA):")
        terms = top_terms[topic_id][:10]  # Show first 10
        for i, term in enumerate(terms, 1):
            print(f"  {i}. {term}")

    # Show NER entities
    if topic_id in vocabulary_dict and vocabulary_dict[topic_id]['ner']:
        print(f"\nNER entities ({len(vocabulary_dict[topic_id]['ner'])} total):")
        ner_sample = vocabulary_dict[topic_id]['ner'][:10]  # Show first 10
        for i, entity in enumerate(ner_sample, 1):
            print(f"  {i}. {entity}")
        if len(vocabulary_dict[topic_id]['ner']) > 10:
            print(f"  ... and {len(vocabulary_dict[topic_id]['ner']) - 10} more")

    # Show DBPedia entities
    if topic_id in vocabulary_dict and vocabulary_dict[topic_id]['dbpedia']:
        print(f"\nDBPedia entities ({len(vocabulary_dict[topic_id]['dbpedia'])} total):")
        dbpedia_sample = vocabulary_dict[topic_id]['dbpedia'][:10]  # Show first 10
        for i, entity in enumerate(dbpedia_sample, 1):
            print(f"  {i}. {entity}")
        if len(vocabulary_dict[topic_id]['dbpedia']) > 10:
            print(f"  ... and {len(vocabulary_dict[topic_id]['dbpedia']) - 10} more")


def create_output_directory(repo_name: str, clustering_method: str = 'agglomerative') -> str:
    """
    Creates the output directory based on clustering method
    """
    if clustering_method == 'spectral':
        output_dir = f'../data/explanations_eng_spectral/{repo_name}'
    else:
        output_dir = f'../data/explanations_eng/{repo_name}'
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def get_user_vocabulary_choice(topic_id: int, vocabulary_dict: Dict[int, Dict[str, List[str]]],
                               top_terms: Dict[int, List[str]]) -> List[str]:
    """
    Allows user to choose vocabulary to use
    """
    print(f"\nVocabulary options for topic {topic_id}:")
    print("1. Use default vocabulary (LDA terms)")
    print("2. Use NER entities")
    print("3. Use DBPedia entities")
    print("4. Combine LDA + NER + DBPedia")
    print("5. Enter custom vocabulary")

    while True:
        try:
            choice = int(input("\nSelect an option (1-5): "))
            if choice in [1, 2, 3, 4, 5]:
                break
            else:
                print("Please select a number between 1 and 5.")
        except ValueError:
            print("Please enter a valid number.")

    vocabulary = []

    if choice == 1:
        # Default vocabulary (LDA terms)
        if topic_id in top_terms:
            vocabulary = top_terms[topic_id][:20]  # Top 20 terms
        print(f"✅ Using LDA vocabulary ({len(vocabulary)} terms)")

    elif choice == 2:
        # NER entities
        if topic_id in vocabulary_dict:
            vocabulary = vocabulary_dict[topic_id]['ner']
        print(f"✅ Using NER entities ({len(vocabulary)} terms)")

    elif choice == 3:
        # DBPedia entities
        if topic_id in vocabulary_dict:
            vocabulary = vocabulary_dict[topic_id]['dbpedia']
        print(f"✅ Using DBPedia entities ({len(vocabulary)} terms)")

    elif choice == 4:
        # Combine all
        if topic_id in top_terms:
            vocabulary.extend(top_terms[topic_id][:10])
        if topic_id in vocabulary_dict:
            vocabulary.extend(vocabulary_dict[topic_id]['ner'])
            vocabulary.extend(vocabulary_dict[topic_id]['dbpedia'])
        vocabulary = list(set(vocabulary))  # Remove duplicates
        print(f"✅ Using combined vocabulary ({len(vocabulary)} terms)")

    elif choice == 5:
        # Custom vocabulary
        print("Enter terms separated by commas:")
        custom_terms = input("Vocabulary: ").strip()
        vocabulary = [term.strip() for term in custom_terms.split(',') if term.strip()]
        print(f"✅ Using custom vocabulary ({len(vocabulary)} terms)")

    return vocabulary


def get_default_vocabulary(topic_id: int, vocabulary_dict: Dict[int, Dict[str, List[str]]],
                           top_terms: Dict[int, List[str]]) -> List[str]:
    """
    Returns default vocabulary (LDA + NER + DBPedia combined) for a topic
    """
    vocabulary = []

    # Add LDA terms
    if topic_id in top_terms:
        vocabulary.extend(top_terms[topic_id][:10])

    # Add NER and DBPedia
    if topic_id in vocabulary_dict:
        vocabulary.extend(vocabulary_dict[topic_id]['ner'])
        vocabulary.extend(vocabulary_dict[topic_id]['dbpedia'])

    # Remove duplicates
    vocabulary = list(set(vocabulary))

    return vocabulary


# ======== MODEL CONFIGURATION ========
SPACY_MODEL = 'en_core_web_lg'
GEN_MODEL = 'Qwen/Qwen2-7B-Instruct'
EVAL_MODEL = 'google/flan-t5-base'
TOP_K_TERMS = 10
N_SINONIMOS = 1
VISITAR_OBJETO = True


# Helper functions
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
    """Extracts JSON from text more robustly"""
    try:
        # Find first { and last }
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            json_str = text[start:end + 1]
            return json.loads(json_str)
    except:
        pass

    # If it fails, try to extract values manually
    try:
        # Search for patterns like "explanation": "text"
        explanation_match = re.search(r'"explanation":\s*"([^"]*)"', text)
        coherence_match = re.search(r'"coherence":\s*(\d+)', text)
        relevance_match = re.search(r'"relevance":\s*(\d+)', text)
        coverage_match = re.search(r'"coverage":\s*(\d+)', text)

        result = {}
        if explanation_match:
            result['explanation'] = explanation_match.group(1)
        if coherence_match:
            result['coherence'] = int(coherence_match.group(1))
        if relevance_match:
            result['relevance'] = int(relevance_match.group(1))
        if coverage_match:
            result['coverage'] = int(coverage_match.group(1))

        if result:
            return result
    except:
        pass

    return None


def setup_configuration():
    """
    Initial system configuration
    """
    print("🚀 Topic Analysis and Explanation System")
    print("=" * 60)

    # 1. Request dataset/repo name
    repo_name = input("Enter dataset/repository name (e.g. amazon, bbc): ").strip()

    if not repo_name:
        print("❌ Error: You must enter a repository name.")
        return None

    try:
        # 2. Load processed dataframe
        print(f"\nLoading data for repository '{repo_name}'...")
        df = load_processed_dataframe(repo_name)
        print(f"✅ Dataframe loaded: {len(df)} rows")

        # 3. Create vocabulary dictionaries
        print("Processing NER and DBPedia vocabulary...")
        vocabulary_dict = create_vocabulary_dictionaries(df)
        print(f"✅ Vocabulary processed for {len(vocabulary_dict)} topics")

        # 4. Load most relevant terms by topic
        print("Loading most relevant terms by topic...")
        top_terms = load_top_terms_by_topic(repo_name)
        print(f"✅ Terms loaded for {len(top_terms)} topics")

        # 5. Create output directory
        output_dir = create_output_directory(repo_name)
        print(f"Output directory: {output_dir}")

        # 6. Show available topics
        available_topics = list(set(list(vocabulary_dict.keys()) + list(top_terms.keys())))
        available_topics.sort()

        print(f"\nAvailable topics: {available_topics}")

        # 7. Request analysis mode: single topic or all topics
        print("\n" + "=" * 60)
        print("ANALYSIS MODE SELECTION")
        print("=" * 60)
        print("1. Analyze a specific topic")
        print("2. Analyze all topics")

        while True:
            try:
                mode_choice = int(input("\nSelect analysis mode (1-2): "))
                if mode_choice in [1, 2]:
                    break
                else:
                    print("Please select 1 or 2.")
            except ValueError:
                print("❌ Please enter a valid number.")

        analyze_all = (mode_choice == 2)
        topic_id = None
        selected_vocabulary = None

        if not analyze_all:
            # Single topic analysis - original behavior
            while True:
                try:
                    topic_id = int(
                        input(
                            f"\nWhich topic do you want to analyze? ({min(available_topics)}-{max(available_topics)}): "))
                    if topic_id in available_topics:
                        break
                    else:
                        print(f"❌ Topic {topic_id} not available. Options: {available_topics}")
                except ValueError:
                    print("❌ Please enter a valid number.")

            # 8. Show topic vocabulary
            show_topic_vocabulary(topic_id, vocabulary_dict, top_terms)

            # 9. Allow user to choose vocabulary
            selected_vocabulary = get_user_vocabulary_choice(topic_id, vocabulary_dict, top_terms)
        else:
            # All topics analysis
            print(f"\n✓ All topics will be analyzed: {available_topics}")
            print("Using default vocabulary (LDA + NER + DBPedia) for all topics...")

        # 10. Define triples path
        triples_path = f'../data/triples_raw/{repo_name}/dataset_triplet_{repo_name}_new_simplificado.csv'
        print(f"\nTriples path: {triples_path}")

        if not os.path.exists(triples_path):
            print(f"⚠️  Warning: Triples file not found: {triples_path}")
            # Search for alternative file
            alt_path = f'../data/triples_raw/processed/dataset_final_triplet_{repo_name}_pykeen'
            if os.path.exists(alt_path):
                triples_path = alt_path
                print(f"✅ Using alternative file: {triples_path}")
            else:
                print("❌ No valid triples file found")
                return None
        else:
            print("✅ Triples file found")

        # 11. Configuration for analysis
        config = {
            'repo_name': repo_name,
            'topic_id': topic_id,
            'vocabulary': selected_vocabulary,
            'triples_path': triples_path,
            'output_dir': output_dir,
            'vocabulary_dict': vocabulary_dict,
            'top_terms': top_terms,
            'analyze_all': analyze_all,
            'available_topics': available_topics,
            'vocabulary_stats': {
                'total_terms': len(selected_vocabulary) if selected_vocabulary else 0,
                'ner_available': len(vocabulary_dict.get(topic_id, {}).get('ner', [])) if topic_id else 0,
                'dbpedia_available': len(vocabulary_dict.get(topic_id, {}).get('dbpedia', [])) if topic_id else 0,
                'lda_available': len(top_terms.get(topic_id, [])) if topic_id else 0
            }
        }

        if not analyze_all:
            config_path = os.path.join(output_dir, f'config_topic_{topic_id}.json')
            with open(config_path, 'w', encoding='utf-8') as f:
                # Serialize only basic data for JSON
                json_config = {k: v for k, v in config.items()
                               if k not in ['vocabulary_dict', 'top_terms', 'available_topics']}
                json.dump(json_config, f, indent=2, ensure_ascii=False)

            print(f"\nConfiguration saved to: {config_path}")
            print(f"\nConfiguration completed. Selected vocabulary: {len(selected_vocabulary)} terms")
        else:
            print(f"\nConfiguration completed for all topics analysis.")

        print("Starting triples analysis and clustering...")

        return config

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Verify that the files exist at the specified paths.")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None


def process_single_topic(topic_id, config, selected_vocabulary, nlp, df_tr, gen, torch_dev, auto_clusters=True,
                         clustering_method='agglomerative'):
    """
    Process a single topic: clustering, explanation generation, and evaluation.
    Returns average metrics (coherence, relevance, coverage) for this topic.

    Args:
        topic_id: ID of the topic to process
        config: Configuration dictionary
        selected_vocabulary: List of vocabulary terms for this topic
        nlp: Spacy NLP model
        df_tr: DataFrame with triplets
        gen: Generation pipeline (can be None)
        torch_dev: Torch device
        auto_clusters: If True, use automatic cluster selection (for batch processing)
        clustering_method: 'agglomerative' or 'spectral' (default: 'agglomerative')

    Returns:
        Dictionary with average metrics and cluster count
    """
    print(f"\n{'=' * 60}")
    print(f"PROCESSING TOPIC {topic_id}")
    print(f"{'=' * 60}")

    TOPIC_ID = topic_id
    TERMINOS_A_INCLUIR = set([term.lower() for term in selected_vocabulary])
    TRIPLES_PATH = config['triples_path']
    OUTPUT_DIR = config['output_dir']

    # Create specific output directory for this topic
    topic_output_dir = os.path.join(OUTPUT_DIR, f'topic_{TOPIC_ID}')
    os.makedirs(topic_output_dir, exist_ok=True)

    # Create compatibility dictionaries with original format
    vocabulary_dict = config['vocabulary_dict']
    if TOPIC_ID in vocabulary_dict:
        # Simulate original DBPedia and NER structure
        dictdbp = {}
        dictner = {}

        # Create dictdbp from extracted DBPedia entities
        for term in vocabulary_dict[TOPIC_ID]['dbpedia']:
            dictdbp[term.lower()] = [{'URI': f'http://dbpedia.org/resource/{term}', 'tipos': []}]

        # Create dictner from extracted NER entities
        for term in vocabulary_dict[TOPIC_ID]['ner']:
            dictner[term.lower()] = 'ENTITY'

        # IMPORTANT: Also add selected LDA vocabulary terms to dictdbp
        for term in selected_vocabulary:
            term_lower = term.lower()
            if term_lower not in dictdbp:
                dictdbp[term_lower] = [{'URI': f'http://dbpedia.org/resource/{term}', 'tipos': []}]
    else:
        dictdbp = {}
        dictner = {}
        # As fallback, create dictdbp from selected vocabulary
        for term in selected_vocabulary:
            term_lower = term.lower()
            dictdbp[term_lower] = [{'URI': f'http://dbpedia.org/resource/{term}', 'tipos': []}]

    print(f"DBPedia dictionary created: {len(dictdbp)} terms")
    print(f"NER dictionary created: {len(dictner)} terms")

    # 1. Filtering and extraction of relevant triplets
    print("\nProcessing relevant triplets...")
    listado_tripletas = []
    palabrasdbpedia = set(k.lower() for k in dictdbp.keys())

    anterior = None
    processed_count = 0
    terms_found_count = 0
    matches_found = 0

    for i, row in df_tr.iterrows():
        if processed_count % 5000 == 0:
            print(f"  Processing: {processed_count}/{len(df_tr)} (terms found: {terms_found_count})")
        processed_count += 1

        tripleta = Tripleta({'subject': str(row['subject']),
                             'relation': row['relation'],
                             'object': str(row['object'])})

        sujeto = set([word.lower() for word in tripleta.sujeto.split()])
        objeto = set([word.lower() for word in tripleta.objeto.split()]) if VISITAR_OBJETO else set()

        if anterior is None:
            anterior = tripleta

        misma_super = (tripleta.esTripletaSuper(anterior) == anterior.esTripletaSuper(tripleta))
        dif = tripleta.dondeSonDiferentes(anterior)

        if (misma_super and (dif == ('sujeto', 'relacion', 'objeto') or dif == ('sujeto', None, 'objeto'))):
            anterior = tripleta
        else:
            continue

        # Verify intersection with selected vocabulary
        has_match = (TERMINOS_A_INCLUIR is None
                     or not TERMINOS_A_INCLUIR.isdisjoint(sujeto)
                     or (VISITAR_OBJETO and not TERMINOS_A_INCLUIR.isdisjoint(objeto)))

        if has_match:
            matches_found += 1
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
                try:
                    for syn in wn.synsets(termino):
                        sinonimos.extend(syn.lemma_names())
                        for h in syn.hypernyms():
                            lwordnet.extend(h.lemma_names())
                except:
                    pass

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
                terms_found_count += 1

    print(f"✅ Triplets processed: {len(listado_tripletas)} relevant terms found")

    if len(listado_tripletas) == 0:
        print("❌ No relevant terms found. Skipping this topic.")
        return {'avg_coherence': 0, 'avg_relevance': 0, 'avg_coverage': 0, 'num_clusters': 0}

    # 2. Expand vocabulary with SpaCy similarity
    print("\nExpanding vocabulary with semantic similarities...")
    df = pd.DataFrame(listado_tripletas)
    vocab_aux, lista_tipos = [], []

    for _, row in df.iterrows():
        termino = row['termino']
        tipos = []

        # More robust type handling
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

        try:
            sims = [nlp(termino).similarity(nlp(t2)) for t2 in tipos_clean]
            if not sims:
                continue

            idx = list(np.argpartition(sims, -N_SINONIMOS)[-N_SINONIMOS:])
            sel = [tipos_clean[i] for i in idx]
            puntuaciones = [sims[i] for i in idx]

            # Correct handling of most similar types
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
        except Exception as e:
            continue

    # Vocabulary lemmatization
    print("Lemmatizing final vocabulary...")
    vocab = set()
    for doc in nlp.pipe(vocab_aux):
        lemmatized = " ".join([token.lemma_.lower() for token in doc])
        vocab.add(lemmatized)

    terms = list(vocab)
    print(f"✅ Final vocabulary: {len(terms)} unique terms")

    if len(terms) < 2:
        print("❌ Error: Not enough terms for clustering")
        return {'avg_coherence': 0, 'avg_relevance': 0, 'avg_coverage': 0, 'num_clusters': 0}

    # Calculate similarity matrix
    print("Calculating similarity matrix...")
    M = np.array([[nlp(t1).similarity(nlp(t2)) for t2 in terms] for t1 in terms])

    # 3. Clustering method selection and optimization

    # Select clustering method
    # Use the parameter value as default, can be overridden in interactive mode
    if auto_clusters:
        # In auto mode, use the method specified in the function parameter
        print(f"✓ Using {clustering_method.capitalize()} clustering (automatic mode)")
    else:
        # In interactive mode, let user choose
        print("\n" + "=" * 60)
        print("CLUSTERING METHOD SELECTION")
        print("=" * 60)
        print("\nSelect clustering method:")
        print("1. Agglomerative (Hierarchical)")
        print("2. Spectral")

        method_choice = input("\nSelect method (1 or 2) [1]: ").strip()

        if method_choice == '2':
            clustering_method = 'spectral'
            print("✓ Using Spectral Clustering")
        else:
            clustering_method = 'agglomerative'
            print("✓ Using Agglomerative (Hierarchical) Clustering")

    print(f"\nOptimizing number of clusters with {clustering_method} method...")
    range_n_clusters = range(2, min(len(terms), 30))
    valores_medios_silhouette = []

    for n_clusters in range_n_clusters:
        if clustering_method == 'spectral':
            # Convert similarity matrix to affinity matrix for spectral clustering
            # Spectral clustering needs positive affinities
            affinity_matrix = (M + 1) / 2  # Convert from [-1,1] to [0,1]
            modelo = SpectralClustering(
                n_clusters=n_clusters,
                affinity='precomputed',
                random_state=42
            )
            labels_temp = modelo.fit_predict(affinity_matrix)
        else:
            # Agglomerative clustering
            modelo = AgglomerativeClustering(
                metric='euclidean',
                linkage='ward',
                n_clusters=n_clusters
            )
            labels_temp = modelo.fit_predict(M)

        silhouette_avg = silhouette_score(M, labels_temp)
        valores_medios_silhouette.append(silhouette_avg)

    optimal_clusters = range_n_clusters[np.argmax(valores_medios_silhouette)]

    # Graphics
    plt.figure(figsize=(10, 6))
    plt.plot(range_n_clusters, valores_medios_silhouette, marker='o')
    plt.axvline(x=optimal_clusters, color='r', linestyle='--', label=f'Automatic Optimum: {optimal_clusters}')
    plt.title(f'Silhouette Index Evolution ({clustering_method.capitalize()}) - Topic {TOPIC_ID}')
    plt.xlabel('Number of clusters')
    plt.ylabel('Average Silhouette')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(topic_output_dir, 'silhouette_evolution.png'))
    plt.close()

    # Create model for dendrogram (only for agglomerative)
    linkage_matrix = None
    if clustering_method == 'agglomerative':
        linkage_matrix = shc.linkage(M, method='ward')

    # Cluster selection
    if auto_clusters:
        final_clusters = optimal_clusters
        print(f"\n✓ Using automatic optimum: {final_clusters} clusters")
    else:
        # Interactive mode
        print("\n" + "=" * 60)
        print("CLUSTERING ANALYSIS")
        print("=" * 60)
        print(f"\nFinal vocabulary ({len(terms)} terms):")
        for i, term in enumerate(terms):
            print(f"  {i:2d}. {term}")

        print(f"\nOptimal number of clusters according to Silhouette: {optimal_clusters}")
        print(f"Maximum Silhouette score: {max(valores_medios_silhouette):.3f}")

        print("\nDo you want to modify the number of clusters?")
        print("(Press Enter to use automatic optimum or enter a number)")

        user_input = input(f"Number of clusters [{optimal_clusters}]: ").strip()

        if user_input and user_input.isdigit():
            user_clusters = int(user_input)
            if 2 <= user_clusters <= len(terms):
                final_clusters = user_clusters
                print(f"\n✓ Using {final_clusters} clusters according to user selection")
            else:
                print(f"\n⚠ Invalid number. Using automatic optimum: {optimal_clusters}")
                final_clusters = optimal_clusters
        else:
            final_clusters = optimal_clusters
            print(f"\n✓ Using automatic optimum number: {final_clusters}")

    # Generate dendrogram with cut line (only for agglomerative)
    if clustering_method == 'agglomerative' and linkage_matrix is not None:
        plt.figure(figsize=(12, 8))

        # Calculate cut height for selected number of clusters
        if final_clusters < len(terms):
            cut_height = linkage_matrix[-(final_clusters - 1), 2] * 0.9
        else:
            cut_height = 0

        dend = shc.dendrogram(linkage_matrix, labels=terms, color_threshold=cut_height)

        # Add horizontal line at cut height
        if cut_height > 0:
            plt.axhline(y=cut_height, color='r', linestyle='--', linewidth=2,
                        label=f'Cut for {final_clusters} clusters')

        plt.title(f"Hierarchical Dendrogram (Ward) - Topic {TOPIC_ID} - {final_clusters} clusters")
        plt.xlabel('Terms')
        plt.ylabel('Distance')
        plt.xticks(rotation=90)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(topic_output_dir, 'dendrogram_with_cut.png'))
        plt.close()

    # Final clustering with selected number
    if clustering_method == 'spectral':
        # Spectral clustering
        affinity_matrix = (M + 1) / 2  # Convert from [-1,1] to [0,1]
        labels = SpectralClustering(
            n_clusters=final_clusters,
            affinity='precomputed',
            random_state=42
        ).fit_predict(affinity_matrix)
    else:
        # Agglomerative clustering
        labels = AgglomerativeClustering(n_clusters=final_clusters, linkage='ward').fit_predict(M)
    sample_sil = silhouette_samples(M, labels)
    global_sil = silhouette_score(M, labels)

    print(f"\nSilhouette score for {final_clusters} clusters: {global_sil:.3f}")

    # Generate UMAP visualization for Spectral clustering
    if clustering_method == 'spectral' and UMAP_AVAILABLE:
        print("\nGenerating UMAP visualization for Spectral clustering...")
        try:
            # Create UMAP embedding
            umap_reducer = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
            embedding = umap_reducer.fit_transform(M)

            # Create visualization
            plt.figure(figsize=(12, 8))
            scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels,
                                  cmap='viridis', s=100, alpha=0.6)
            plt.colorbar(scatter, label='Cluster')

            # Add term labels to points
            for i, term in enumerate(terms):
                plt.annotate(term, (embedding[i, 0], embedding[i, 1]),
                             fontsize=8, alpha=0.7)

            plt.title(f'UMAP Visualization - Spectral Clustering - Topic {TOPIC_ID}')
            plt.xlabel('UMAP 1')
            plt.ylabel('UMAP 2')
            plt.tight_layout()
            plt.savefig(os.path.join(topic_output_dir, 'umap_spectral_clusters.png'), dpi=150)
            plt.close()
            print("✅ UMAP visualization saved")

            # Save embedding data for later analysis
            np.save(os.path.join(topic_output_dir, 'umap_embedding.npy'), embedding)

        except Exception as e:
            print(f"⚠️ Could not generate UMAP visualization: {e}")
    elif clustering_method == 'spectral' and not UMAP_AVAILABLE:
        print("⚠️ UMAP not available. Skipping UMAP visualization.")

    clusters = {}
    for cl in set(labels):
        idxs = np.where(labels == cl)[0]
        term_sils = [(terms[i], sample_sil[i]) for i in idxs]
        top_terms = [t for t, _ in sorted(term_sils, key=lambda x: -x[1])[:TOP_K_TERMS]]
        clusters[cl] = top_terms

    # Show cluster summary
    print("\nCLUSTER SUMMARY:")
    print("-" * 60)
    for cl, terms_cl in clusters.items():
        print(f"Cluster {cl}: {', '.join(terms_cl)}")

    clusters_str = {str(k): v for k, v in clusters.items()}

    with open(os.path.join(topic_output_dir, 'clusters.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'topic_id': TOPIC_ID,
            'clustering_method': clustering_method,
            'best_k': len(clusters_str),
            'selected_k': final_clusters,
            'automatic_optimal_k': optimal_clusters,
            'global_sil': global_sil,
            'clusters': clusters_str
        }, f, ensure_ascii=False, indent=2)

    # 4. GENERATION WITH ADVANCED PROMPTING STRATEGIES
    def generate_explanation_with_cot(cluster_id, terms, topic_id, silhouette_score):
        """Chain-of-Thought strategy for explanation generation"""
        if gen is None:
            return "Explanation not available - model not loaded", "", False

        # Step 1: Key phrase extraction
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

        try:
            key_phrases_output = gen(key_phrase_prompt, max_new_tokens=100, temperature=0.1)
            key_phrases_text = key_phrases_output[0].get('generated_text', '').strip()
        except:
            key_phrases_text = f"semantic cluster, {terms[0]} related"

        # Step 2: Relevance verification
        verification_prompt = f"""VERIFICATION TASK: Check if these key phrases accurately represent the given terms.

TERMS: {', '.join(terms)}
EXTRACTED KEY PHRASES: {key_phrases_text}

VERIFICATION CRITERIA:
1. Do the key phrases accurately reflect the semantic relationships?
2. Are they relevant to the technology domain?
3. Do they avoid hallucinated connections?

VERIFICATION RESULT (True/False): """

        try:
            verification_output = gen(verification_prompt, max_new_tokens=50, temperature=0.1)
            verification_text = verification_output[0].get('generated_text', '').strip()
            is_verified = 'true' in verification_text.lower()
        except:
            is_verified = True

        # Step 3: Final explanation generation
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
    "explanation": "Clear, concise explanation of the cluster's semantic unity",
    "coherence": [1-5 score],
    "relevance": [1-5 score],
    "coverage": [1-5 score],
    "key_phrases": [list of verified key phrases],
    "reasoning": "Brief justification for the scores"
}}

JSON Response:"""

        try:
            explanation_output = gen(explanation_prompt, max_new_tokens=250, temperature=0.2)
            explanation_text = explanation_output[0].get('generated_text', '')
        except:
            explanation_text = f'{{"explanation": "Technology cluster: {", ".join(terms[:3])}", "coherence": 3, "relevance": 3, "coverage": 3}}'

        return explanation_text, key_phrases_text, is_verified

    # Generate explanations with advanced strategy
    print("\n✨ Generating explanations with Chain-of-Thought...")
    explanations = {}
    detailed_analysis = {}

    for cid, terms_c in clusters.items():
        try:
            print(f"  Processing cluster {cid}...")
            explanation_text, key_phrases, verified = generate_explanation_with_cot(
                cid, terms_c, TOPIC_ID, global_sil
            )

            # Extract JSON from explanation
            parsed = extract_json_from_text(explanation_text)
            if parsed is None:
                parsed = {
                    'explanation': f"Technology cluster grouping: {', '.join(terms_c[:3])}",
                    'coherence': 3,
                    'relevance': 3,
                    'coverage': 3,
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
            print(f"  ❌ Error generating explanation for cluster {cid}: {e}")
            explanations[cid] = {
                'explanation': f"Technology cluster: {', '.join(terms_c[:3])}",
                'coherence': 3,
                'relevance': 3,
                'coverage': 3,
                'key_phrases': [],
                'reasoning': f'Error in generation: {str(e)[:50]}'
            }

    explanations_str = {str(k): v for k, v in explanations.items()}
    with open(os.path.join(topic_output_dir, 'explanations.json'), 'w', encoding='utf-8') as f:
        json.dump(explanations_str, f, ensure_ascii=False, indent=2)

    with open(os.path.join(topic_output_dir, 'detailed_analysis.json'), 'w', encoding='utf-8') as f:
        json.dump({str(k): v for k, v in detailed_analysis.items()}, f, ensure_ascii=False, indent=2)

    # 5. EVALUATION WITH XAI SPECIFIC CRITERIA
    def evaluate_explanation_structured(cluster_id, explanation, terms, topic_id):
        """Structured evaluation based on specific XAI criteria"""
        evaluation_prompt = f"""You are an expert evaluator of AI explanations. Evaluate this cluster explanation using XAI criteria.

CLUSTER INFORMATION:
- ID: {topic_id}-{cluster_id}
- Terms: {', '.join(terms)}
- Explanation: "{explanation.get('explanation', '')}"
- Generated Key Phrases: {explanation.get('key_phrases', [])}

EVALUATION CRITERIA (score 1-5):

COHERENCE (Semantic Coherence):
- Are the terms semantically related?
- Does the explanation capture their unity?
- Is the clustering logically sound?

RELEVANCE (Domain Relevance):
- How relevant are these terms to the technology domain?
- Does the explanation connect to the broader topic context?
- Are domain-specific relationships identified?

COVERAGE (Coverage Completeness):
- Does the explanation cover the main aspects of the cluster?
- Are key semantic relationships addressed?
- Is the scope appropriate for the term set?

PROVIDE EVALUATION as JSON:
{{
    "coherence": [1-5],
    "relevance": [1-5],
    "coverage": [1-5],
    "justification": "2-sentence explanation of scores",
    "strengths": ["strength1", "strength2"],
    "weaknesses": ["weakness1", "weakness2"]
}}

JSON Evaluation:"""
        return evaluation_prompt

    # Perform structured evaluation
    print("\nEvaluating explanations with XAI criteria...")

    evaluations = {}
    for cid, exp in explanations.items():
        terms_c = clusters[cid]
        print(f"  Evaluating cluster {cid}...")

        eval_prompt = evaluate_explanation_structured(cid, exp, terms_c, TOPIC_ID)

        try:
            if gen is not None:
                eval_output = gen(eval_prompt, max_new_tokens=200, temperature=0.1)
                eval_text = eval_output[0].get('generated_text', '')
                parsed = extract_json_from_text(eval_text)
            else:
                parsed = None

            if parsed is None:
                parsed = {
                    'coherence': 3,
                    'relevance': 3,
                    'coverage': 3,
                    'justification': 'Technology terms related with medium coherence',
                    'strengths': ['Clear semantic grouping'],
                    'weaknesses': ['Needs greater specificity']
                }
        except Exception as e:
            parsed = {
                'coherence': 3,
                'relevance': 3,
                'coverage': 3,
                'justification': 'Automatic evaluation with error',
                'strengths': ['Basic grouping'],
                'weaknesses': ['Processing error']
            }

        evaluations[cid] = parsed

    evaluations_str = {str(k): v for k, v in evaluations.items()}
    with open(os.path.join(topic_output_dir, 'evaluations.json'), 'w', encoding='utf-8') as f:
        json.dump(evaluations_str, f, ensure_ascii=False, indent=2)

    # 6. EXECUTIVE SUMMARY
    print("\nGenerating executive summary...")

    with open(os.path.join(topic_output_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write(f"TOPIC ANALYSIS - EXECUTIVE REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"ANALYSIS CONFIGURATION\n")
        f.write(f"Repository: {config['repo_name']}\n")
        f.write(f"Topic ID: {TOPIC_ID}\n")
        f.write(f"Selected vocabulary: {len(selected_vocabulary)} terms\n")
        f.write(f"Number of identified clusters: {len(clusters)}\n")
        f.write(f"Automatic optimal number: {optimal_clusters}\n")
        f.write(f"Number selected: {final_clusters}\n")
        f.write(f"Clustering quality (Silhouette): {global_sil:.3f}\n")
        f.write(f"Method: {clustering_method.capitalize()} clustering + Explanatory LLM\n\n")

        f.write("SUMMARY BY CLUSTER:\n")
        f.write("-" * 40 + "\n\n")

        for cid, terms_c in clusters.items():
            exp = explanations[cid]
            eva = evaluations[cid]

            f.write(f"CLUSTER {cid}:\n")
            f.write(f"  Key terms: {', '.join(terms_c)}\n")
            f.write(f"  \n")
            f.write(f"  Explanation: {exp.get('explanation', 'N/A')}\n")
            f.write(f"  \n")
            f.write(f"  Scores:\n")
            f.write(f"     • Coherence: {eva.get('coherence', 'N/A')}/5\n")
            f.write(f"     • Relevance: {eva.get('relevance', 'N/A')}/5\n")
            f.write(f"     • Coverage: {eva.get('coverage', 'N/A')}/5\n")
            f.write(f"  \n")
            f.write(f"  Strengths: {', '.join(eva.get('strengths', ['N/A']))}\n")
            f.write(f"  Weaknesses: {', '.join(eva.get('weaknesses', ['N/A']))}\n")
            f.write(f"  \n")
            f.write(f"  Justification: {eva.get('justification', 'N/A')}\n")
            f.write(f"  \n")

            if cid in detailed_analysis:
                detail = detailed_analysis[cid]
                f.write(f"  Detailed analysis:\n")
                f.write(f"     • Extracted key phrases: {detail.get('key_phrases_extracted', 'N/A')}\n")
                f.write(f"     • Verification passed: {detail.get('verification_passed', 'N/A')}\n")

            f.write("\n" + "-" * 40 + "\n\n")

        # Global statistics
        coherences = [eva.get('coherence', 0) for eva in evaluations.values() if
                      isinstance(eva.get('coherence'), (int, float))]
        relevances = [eva.get('relevance', 0) for eva in evaluations.values() if
                      isinstance(eva.get('relevance'), (int, float))]
        coverages = [eva.get('coverage', 0) for eva in evaluations.values() if
                     isinstance(eva.get('coverage'), (int, float))]

        if coherences:
            f.write("GLOBAL STATISTICS:\n")
            f.write(f"  Average coherence: {np.mean(coherences):.2f}/5\n")
            f.write(f"  Average relevance: {np.mean(relevances):.2f}/5\n")
            f.write(f"  Average coverage: {np.mean(coverages):.2f}/5\n")
            f.write(f"  Clustering quality: {global_sil:.3f}\n\n")

    print("✅ Executive summary generated")

    # Calculate average metrics to return
    coherences = [eva.get('coherence', 0) for eva in evaluations.values() if
                  isinstance(eva.get('coherence'), (int, float))]
    relevances = [eva.get('relevance', 0) for eva in evaluations.values() if
                  isinstance(eva.get('relevance'), (int, float))]
    coverages = [eva.get('coverage', 0) for eva in evaluations.values() if
                 isinstance(eva.get('coverage'), (int, float))]

    avg_coherence = np.mean(coherences) if coherences else 0
    avg_relevance = np.mean(relevances) if relevances else 0
    avg_coverage = np.mean(coverages) if coverages else 0

    print(f"\nTOPIC {TOPIC_ID} COMPLETED!")
    print(f"  Clusters: {len(clusters)}")
    print(f"  Avg Coherence: {avg_coherence:.2f}/5")
    print(f"  Avg Relevance: {avg_relevance:.2f}/5")
    print(f"  Avg Coverage: {avg_coverage:.2f}/5")

    return {
        'topic_id': TOPIC_ID,
        'avg_coherence': avg_coherence,
        'avg_relevance': avg_relevance,
        'avg_coverage': avg_coverage,
        'num_clusters': len(clusters),
        'output_dir': topic_output_dir
    }


# ======== COMPARISON FUNCTIONS ========
def compare_clustering_methods(repo_name: str) -> pd.DataFrame:
    """
    Compare results from hierarchical and spectral clustering methods
    """
    hierarchical_path = f'../data/explanations_eng/{repo_name}/all_topics_summary.json'
    spectral_path = f'../data/explanations_eng_spectral/{repo_name}/all_topics_summary.json'

    comparison_data = []

    # Load hierarchical results if exists
    hierarchical_metrics = {}
    if os.path.exists(hierarchical_path):
        with open(hierarchical_path, 'r') as f:
            hierarchical_data = json.load(f)
            for topic in hierarchical_data.get('topics_metrics', []):
                hierarchical_metrics[topic['topic_id']] = {
                    'coherence': topic['avg_coherence'],
                    'relevance': topic['avg_relevance'],
                    'coverage': topic['avg_coverage'],
                    'num_clusters': topic['num_clusters']
                }
        print(f"✅ Loaded hierarchical results: {len(hierarchical_metrics)} topics")
    else:
        print(f"❌ Hierarchical results not found at {hierarchical_path}")

    # Load spectral results if exists
    spectral_metrics = {}
    if os.path.exists(spectral_path):
        with open(spectral_path, 'r') as f:
            spectral_data = json.load(f)
            for topic in spectral_data.get('topics_metrics', []):
                spectral_metrics[topic['topic_id']] = {
                    'coherence': topic['avg_coherence'],
                    'relevance': topic['avg_relevance'],
                    'coverage': topic['avg_coverage'],
                    'num_clusters': topic['num_clusters']
                }
        print(f"✅ Loaded spectral results: {len(spectral_metrics)} topics")
    else:
        print(f"❌ Spectral results not found at {spectral_path}")

    # Create comparison dataframe
    common_topics = set(hierarchical_metrics.keys()) & set(spectral_metrics.keys())

    for topic_id in common_topics:
        h_metrics = hierarchical_metrics[topic_id]
        s_metrics = spectral_metrics[topic_id]

        comparison_data.append({
            'topic_id': topic_id,
            'hierarchical_coherence': h_metrics['coherence'],
            'spectral_coherence': s_metrics['coherence'],
            'coherence_diff': s_metrics['coherence'] - h_metrics['coherence'],
            'hierarchical_relevance': h_metrics['relevance'],
            'spectral_relevance': s_metrics['relevance'],
            'relevance_diff': s_metrics['relevance'] - h_metrics['relevance'],
            'hierarchical_coverage': h_metrics['coverage'],
            'spectral_coverage': s_metrics['coverage'],
            'coverage_diff': s_metrics['coverage'] - h_metrics['coverage'],
            'hierarchical_clusters': h_metrics['num_clusters'],
            'spectral_clusters': s_metrics['num_clusters'],
            'clusters_diff': s_metrics['num_clusters'] - h_metrics['num_clusters']
        })

    return pd.DataFrame(comparison_data)


def generate_comparison_report(repositories: List[str]) -> None:
    """
    Generate a comprehensive comparison report for all repositories
    """
    print("\n" + "=" * 80)
    print("CLUSTERING METHODS COMPARISON REPORT")
    print("=" * 80)

    all_comparisons = []

    for repo in repositories:
        print(f"\n{repo.upper()} Repository:")
        print("-" * 40)

        df_comparison = compare_clustering_methods(repo)

        if df_comparison.empty:
            print(f"⚠️ No comparison data available for {repo}")
            continue

        all_comparisons.append((repo, df_comparison))

        # Calculate average metrics
        avg_metrics = {
            'hierarchical': {
                'coherence': df_comparison['hierarchical_coherence'].mean(),
                'relevance': df_comparison['hierarchical_relevance'].mean(),
                'coverage': df_comparison['hierarchical_coverage'].mean(),
                'clusters': df_comparison['hierarchical_clusters'].mean()
            },
            'spectral': {
                'coherence': df_comparison['spectral_coherence'].mean(),
                'relevance': df_comparison['spectral_relevance'].mean(),
                'coverage': df_comparison['spectral_coverage'].mean(),
                'clusters': df_comparison['spectral_clusters'].mean()
            }
        }

        # Print comparison table
        print("\nAverage Metrics Comparison:")
        print(f"{'Metric':<15} {'Hierarchical':<15} {'Spectral':<15} {'Difference':<15}")
        print("-" * 60)

        for metric in ['coherence', 'relevance', 'coverage']:
            h_val = avg_metrics['hierarchical'][metric]
            s_val = avg_metrics['spectral'][metric]
            diff = s_val - h_val
            sign = '+' if diff > 0 else ''
            print(f"{metric.capitalize():<15} {h_val:<15.3f} {s_val:<15.3f} {sign}{diff:<15.3f}")

        # Clusters comparison
        h_clusters = avg_metrics['hierarchical']['clusters']
        s_clusters = avg_metrics['spectral']['clusters']
        diff_clusters = s_clusters - h_clusters
        sign = '+' if diff_clusters > 0 else ''
        print(f"{'Avg Clusters':<15} {h_clusters:<15.1f} {s_clusters:<15.1f} {sign}{diff_clusters:<15.1f}")

        # Determine winner
        print("\n📊 Method Performance:")
        better_method = []
        for metric in ['coherence', 'relevance', 'coverage']:
            if avg_metrics['spectral'][metric] > avg_metrics['hierarchical'][metric]:
                better_method.append('spectral')
                print(
                    f"  ✓ Spectral better for {metric} (+{avg_metrics['spectral'][metric] - avg_metrics['hierarchical'][metric]:.3f})")
            elif avg_metrics['hierarchical'][metric] > avg_metrics['spectral'][metric]:
                better_method.append('hierarchical')
                print(
                    f"  ✓ Hierarchical better for {metric} (+{avg_metrics['hierarchical'][metric] - avg_metrics['spectral'][metric]:.3f})")
            else:
                print(f"  = Equal performance for {metric}")

        # Overall winner
        if better_method.count('spectral') > better_method.count('hierarchical'):
            print(f"\n🏆 Overall Winner: SPECTRAL CLUSTERING")
        elif better_method.count('hierarchical') > better_method.count('spectral'):
            print(f"\n🏆 Overall Winner: HIERARCHICAL CLUSTERING")
        else:
            print(f"\n🤝 Overall Result: TIE")

        # Save detailed comparison
        comparison_path = f'../data/comparison_{repo}_methods.csv'
        df_comparison.to_csv(comparison_path, index=False)
        print(f"\n💾 Detailed comparison saved to: {comparison_path}")

    # Generate consolidated report if multiple repositories
    if len(all_comparisons) > 1:
        print("\n" + "=" * 80)
        print("CONSOLIDATED COMPARISON ACROSS ALL REPOSITORIES")
        print("=" * 80)

        consolidated_metrics = {
            'hierarchical': {'coherence': [], 'relevance': [], 'coverage': []},
            'spectral': {'coherence': [], 'relevance': [], 'coverage': []}
        }

        for repo, df in all_comparisons:
            for metric in ['coherence', 'relevance', 'coverage']:
                consolidated_metrics['hierarchical'][metric].append(df[f'hierarchical_{metric}'].mean())
                consolidated_metrics['spectral'][metric].append(df[f'spectral_{metric}'].mean())

        print(f"\n{'Metric':<15} {'Hierarchical (avg)':<20} {'Spectral (avg)':<20}")
        print("-" * 55)

        for metric in ['coherence', 'relevance', 'coverage']:
            h_avg = np.mean(consolidated_metrics['hierarchical'][metric])
            s_avg = np.mean(consolidated_metrics['spectral'][metric])
            print(f"{metric.capitalize():<15} {h_avg:<20.3f} {s_avg:<20.3f}")

        # Save consolidated report
        report_path = '../data/consolidated_comparison_report.json'
        with open(report_path, 'w') as f:
            json.dump({
                'repositories': repositories,
                'hierarchical_metrics': {k: np.mean(v).tolist() for k, v in
                                         consolidated_metrics['hierarchical'].items()},
                'spectral_metrics': {k: np.mean(v).tolist() for k, v in consolidated_metrics['spectral'].items()},
                'timestamp': pd.Timestamp.now().isoformat()
            }, f, indent=2)

        print(f"\n💾 Consolidated report saved to: {report_path}")


def main():
    """
    Enhanced main script function with batch processing and comparison support
    """
    print("\n" + "=" * 80)
    print("ENHANCED CLUSTERING ANALYSIS WITH COMPARISON SUPPORT")
    print("=" * 80)

    # Mode selection
    print("\nExecution modes:")
    print("1. Interactive mode (single repository)")
    print("2. Batch mode (multiple repositories with default method)")
    print("3. Comparison mode (run both methods and compare)")

    mode = input("\nSelect mode (1-3): ").strip()

    if mode == '2' or mode == '3':
        # Batch or comparison mode
        print("\nAvailable repositories: amazon, bbc, reuters_activities")
        repos_input = input("Enter repositories separated by comma (or 'all' for all): ").strip()

        if repos_input.lower() == 'all':
            repositories = ['amazon', 'bbc', 'reuters_activities']
        else:
            repositories = [r.strip() for r in repos_input.split(',')]

        # Select default clustering method
        if mode == '3':
            # Comparison mode - run both methods
            methods_to_run = ['agglomerative', 'spectral']
            print("✓ Will run both methods for comparison")
        else:
            # Batch mode - select one method
            print("\nSelect default clustering method:")
            print("1. Agglomerative (Hierarchical)")
            print("2. Spectral")

            method_choice = input("Select method (1 or 2) [1]: ").strip()
            if method_choice == '2':
                methods_to_run = ['spectral']
            else:
                methods_to_run = ['agglomerative']

        # Process each repository with selected methods
        for repo_name in repositories:
            for clustering_method in methods_to_run:
                print(f"\n" + "=" * 80)
                print(f"PROCESSING: {repo_name.upper()} - {clustering_method.upper()} METHOD")
                print("=" * 80)

                # Load data for this repository
                df = load_processed_dataframe(repo_name)
                vocabulary_dict = create_vocabulary_dictionaries(df)
                available_topics = sorted(vocabulary_dict.keys())
                top_terms = load_top_terms_by_topic(repo_name)

                # Create output directory based on method
                output_dir = create_output_directory(repo_name, clustering_method)

                # Load triplets
                triples_path = f'../data/triples_raw/{repo_name}/dataset_triplet_{repo_name}_new_simplificado.csv'
                if not os.path.exists(triples_path):
                    print(f"❌ Triplets file not found for {repo_name}")
                    continue

                df_tr = pd.read_csv(triples_path)
                print(f"✅ Triplets loaded: {len(df_tr)} rows")

                # Model preparation
                try:
                    nlp = spacy.load(SPACY_MODEL)
                except OSError:
                    print(f"❌ Error: SpaCy model '{SPACY_MODEL}' not found")
                    return

                nltk.download('wordnet', quiet=True)

                # Load generative model
                torch_dev = 'cuda' if torch.cuda.is_available() else 'cpu'
                print(f"Device: {torch_dev}")

                try:
                    tok_g = AutoTokenizer.from_pretrained(GEN_MODEL)
                    mod_g = AutoModelForCausalLM.from_pretrained(GEN_MODEL).to(torch_dev)
                    gen = pipeline('text-generation', model=mod_g, tokenizer=tok_g,
                                   device=0 if torch.cuda.is_available() else -1,
                                   return_full_text=False)
                    print("✅ Generative model loaded")
                except Exception as e:
                    print(f"❌ Error loading generative model: {e}")
                    gen = None

                # Process all topics
                all_topics_metrics = []

                for topic_id in available_topics:
                    # Get default vocabulary for this topic
                    vocab = []
                    if topic_id in vocabulary_dict and 'ner' in vocabulary_dict[topic_id]:
                        vocab.extend(vocabulary_dict[topic_id]['ner'])
                    if topic_id in top_terms:
                        vocab.extend(top_terms[topic_id][:50])
                    if topic_id in vocabulary_dict and 'dbpedia' in vocabulary_dict[topic_id]:
                        vocab.extend(vocabulary_dict[topic_id]['dbpedia'])
                    vocab = list(set([v.lower() for v in vocab if v]))

                    if len(vocab) == 0:
                        print(f"⚠️ No vocabulary for topic {topic_id}, skipping...")
                        continue

                    # Create config for this topic
                    config = {
                        'repo_name': repo_name,
                        'topic_id': topic_id,
                        'vocabulary': vocab,
                        'triples_path': triples_path,
                        'output_dir': output_dir,
                        'vocabulary_dict': vocabulary_dict,
                        'top_terms': top_terms,
                        'available_topics': available_topics
                    }

                    # Process topic with specified clustering method
                    metrics = process_single_topic(
                        topic_id,
                        config,
                        vocab,
                        nlp,
                        df_tr,
                        gen,
                        torch_dev,
                        auto_clusters=True,
                        clustering_method=clustering_method
                    )

                    all_topics_metrics.append(metrics)

                # Save summary for this repository and method
                summary_path = os.path.join(output_dir, 'all_topics_summary.json')
                with open(summary_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'repository': repo_name,
                        'clustering_method': clustering_method,
                        'generation_model': GEN_MODEL,
                        'total_topics': len(available_topics),
                        'topics_processed': len(all_topics_metrics),
                        'topics_metrics': all_topics_metrics
                    }, f, ensure_ascii=False, indent=2)

                print(f"\n✅ Summary saved: {summary_path}")

                # Print summary statistics
                if all_topics_metrics:
                    avg_coherence = np.mean([m['avg_coherence'] for m in all_topics_metrics])
                    avg_relevance = np.mean([m['avg_relevance'] for m in all_topics_metrics])
                    avg_coverage = np.mean([m['avg_coverage'] for m in all_topics_metrics])

                    print(f"\n📊 Overall Statistics for {repo_name} - {clustering_method}:")
                    print(f"  Average Coherence: {avg_coherence:.3f}")
                    print(f"  Average Relevance: {avg_relevance:.3f}")
                    print(f"  Average Coverage: {avg_coverage:.3f}")

        # If comparison mode, generate comparison report
        if mode == '3':
            print("\n" + "=" * 80)
            print("GENERATING COMPARISON REPORT")
            print("=" * 80)
            generate_comparison_report(repositories)

    else:
        # Interactive mode (original behavior)
        config = setup_configuration()
        if config is None:
            return

        # Model preparation
        try:
            nlp = spacy.load(SPACY_MODEL)
        except OSError:
            print(f"❌ Error: SpaCy model '{SPACY_MODEL}' not found")
            return

        nltk.download('wordnet', quiet=True)

        try:
            df_tr = pd.read_csv(config['triples_path'])
            print(f"✅ Triplets loaded: {len(df_tr)} rows")
        except Exception as e:
            print(f"❌ Error loading triplets: {e}")
            return

        # Load generative model
        torch_dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Device: {torch_dev}")

        try:
            tok_g = AutoTokenizer.from_pretrained(GEN_MODEL)
            mod_g = AutoModelForCausalLM.from_pretrained(GEN_MODEL).to(torch_dev)
            gen = pipeline('text-generation', model=mod_g, tokenizer=tok_g,
                           device=0 if torch.cuda.is_available() else -1,
                           return_full_text=False)
            print("✅ Generative model loaded")
        except Exception as e:
            print(f"❌ Error loading generative model: {e}")
            gen = None

        # Process based on configuration
        if config.get('analyze_all', False):
            # Process all topics
            all_topics_metrics = []

            for topic_id in config['available_topics']:
                # Get default vocabulary
                vocab_dict = config['vocabulary_dict']
                top_terms = config['top_terms']

                vocab = []
                if topic_id in vocab_dict and 'ner' in vocab_dict[topic_id]:
                    vocab.extend(vocab_dict[topic_id]['ner'])
                if topic_id in top_terms:
                    vocab.extend(top_terms[topic_id][:50])
                if topic_id in vocab_dict and 'dbpedia' in vocab_dict[topic_id]:
                    vocab.extend(vocab_dict[topic_id]['dbpedia'])
                vocab = list(set([v.lower() for v in vocab if v]))

                if len(vocab) == 0:
                    continue

                metrics = process_single_topic(
                    topic_id,
                    config,
                    vocab,
                    nlp,
                    df_tr,
                    gen,
                    torch_dev,
                    auto_clusters=False,
                    clustering_method='agglomerative'  # Default for interactive
                )

                all_topics_metrics.append(metrics)

            # Save summary
            summary_path = os.path.join(config['output_dir'], 'all_topics_summary.json')
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'repository': config['repo_name'],
                    'generation_model': GEN_MODEL,
                    'total_topics': len(config['available_topics']),
                    'topics_processed': len(all_topics_metrics),
                    'topics_metrics': all_topics_metrics
                }, f, ensure_ascii=False, indent=2)

            print(f"\n✅ Analysis complete. Summary saved to {summary_path}")

        else:
            # Single topic analysis
            process_single_topic(
                config['topic_id'],
                config,
                config['vocabulary'],
                nlp,
                df_tr,
                gen,
                torch_dev,
                auto_clusters=False
            )

            print(f"\n✅ Analysis complete for topic {config['topic_id']}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
