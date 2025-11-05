"""
Post-hoc Clustering Comparison for OBOE Framework
Analyzes existing similarity matrices with multiple clustering strategies

This script:
1. Loads processed data and triplets for each corpus
2. Generates similarity matrices following the original pipeline
3. Compares Ward, K-Means, and Spectral clustering
4. Produces comparison tables for the paper

Author: [Your name]
Date: [Current date]
Purpose: Respond to Reviewer 3 - clustering strategy comparison
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics.pairwise import cosine_distances
import spacy
import joblib
import pickle
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# Import from your existing code
from utils.types import StringCaseInsensitiveSet, CaseInsensitiveDict, CaseInsensitiveSet
from utils.triplet_manager_lib import Tripleta
import nltk
from nltk.corpus import wordnet as wn

# Configuration
SPACY_MODEL = 'en_core_web_lg'
MAX_CLUSTERS = 8  # Maximum number of clusters to evaluate per topic
N_SINONIMOS = 1
VISITAR_OBJETO = True
REPOSITORIES = ['amazon', 'bbc', 'reuters_activities']  # Corpora to analyze


def load_processed_dataframe(repo_name: str) -> pd.DataFrame:
    """Load processed dataframe for repository"""
    processed_path = f'../data/processed/{repo_name}/{repo_name}_processed_semantic.pkl'

    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"File not found: {processed_path}")

    with open(processed_path, 'rb') as f:
        df = joblib.load(f)

    print(f"  ✅ Loaded {len(df)} documents")
    return df


def load_top_terms(repo_name: str) -> dict:
    """Load LDA top terms by topic"""
    terms_path = f'../data/lda_eval/{repo_name}/top_terms_by_topic.pkl'

    if not os.path.exists(terms_path):
        raise FileNotFoundError(f"File not found: {terms_path}")

    with open(terms_path, 'rb') as f:
        top_terms = joblib.load(f)

    return top_terms


def create_vocabulary_dict(df: pd.DataFrame) -> dict:
    """
    Create vocabulary dictionaries from processed dataframe
    Simplified version focusing on DBPedia entities
    """
    vocabulary_dict = {}

    # Detect topic column
    topic_col = None
    for col in df.columns:
        if 'target' in str(col).lower() or 'topic' in str(col).lower():
            topic_col = col
            break

    if topic_col is None:
        print("  ⚠️ Warning: No topic column found, using index")
        df['topic_id'] = df.index
        topic_col = 'topic_id'

    # Detect DBPedia column
    dbpedia_col = None
    for col in df.columns:
        sample = df.iloc[0][col] if len(df) > 0 else None
        if isinstance(sample, dict):
            # Check if has DBPedia structure
            for k, v in sample.items():
                if isinstance(v, dict) and ('URI' in v or 'types' in v):
                    dbpedia_col = col
                    break
            if dbpedia_col:
                break

    print(f"  Detected topic column: {topic_col}")
    print(f"  Detected DBPedia column: {dbpedia_col}")

    for idx, row in df.iterrows():
        topic_id = row[topic_col]

        dbpedia_entities = []

        if dbpedia_col and dbpedia_col in row and row[dbpedia_col]:
            data = row[dbpedia_col]
            if isinstance(data, dict):
                for entity_name, entity_data in data.items():
                    entity_clean = str(entity_name).strip().lower()
                    if entity_clean and len(entity_clean) > 1:
                        dbpedia_entities.append(entity_clean)

        vocabulary_dict[topic_id] = {
            'dbpedia': list(set(dbpedia_entities))
        }

    return vocabulary_dict


def extract_similarity_matrix_for_topic(
        topic_id: int,
        repo_name: str,
        df_triplets: pd.DataFrame,
        vocabulary_dict: dict,
        top_terms: dict,
        nlp
) -> tuple:
    """
    Extract similarity matrix for a topic following original pipeline

    Returns:
        (similarity_matrix, terms, dictdbp) or (None, None, None) if error
    """
    print(f"\n  Processing Topic {topic_id}...")

    # Get vocabulary for this topic
    vocab_lda = top_terms.get(topic_id, [])[:20]
    vocab_dbp = vocabulary_dict.get(topic_id, {}).get('dbpedia', [])

    TERMINOS_A_INCLUIR = set([term.lower() for term in vocab_lda])

    # Create dictdbp
    dictdbp = {}
    for term in vocab_dbp:
        dictdbp[term.lower()] = [{'URI': f'http://dbpedia.org/resource/{term}', 'tipos': []}]

    for term in vocab_lda:
        term_lower = term.lower()
        if term_lower not in dictdbp:
            dictdbp[term_lower] = [{'URI': f'http://dbpedia.org/resource/{term}', 'tipos': []}]

    # Extract triplets (following original logic)
    listado_tripletas = []
    palabrasdbpedia = set(k.lower() for k in dictdbp.keys())

    anterior = None
    terms_found = 0

    for i, row in df_triplets.iterrows():
        tripleta = Tripleta({
            'subject': str(row['subject']),
            'relation': row['relation'],
            'object': str(row['object'])
        })

        sujeto = set([word.lower() for word in tripleta.sujeto.split()])
        objeto = set([word.lower() for word in tripleta.objeto.split()]) if VISITAR_OBJETO else set()

        if anterior is None:
            anterior = tripleta

        misma_super = (tripleta.esTripletaSuper(anterior) == anterior.esTripletaSuper(tripleta))
        dif = tripleta.dondeSonDiferentes(anterior)

        if not (misma_super and (dif == ('sujeto', 'relacion', 'objeto') or dif == ('sujeto', None, 'objeto'))):
            continue

        anterior = tripleta

        # Check vocabulary intersection
        has_match = (TERMINOS_A_INCLUIR is None
                     or not TERMINOS_A_INCLUIR.isdisjoint(sujeto)
                     or (VISITAR_OBJETO and not TERMINOS_A_INCLUIR.isdisjoint(objeto)))

        if not has_match:
            continue

        visitados = set()
        encontradas = sujeto.intersection(palabrasdbpedia)

        if VISITAR_OBJETO:
            encontradas.update(objeto.intersection(palabrasdbpedia))

        for termino in encontradas:
            if termino in visitados or termino[0].isdigit():
                continue

            info_list = dictdbp.get(termino.lower(), [])
            if not info_list:
                continue

            info_termino = info_list[0]

            # WordNet synonyms
            sinonimos = []
            lwordnet = []
            try:
                for syn in wn.synsets(termino):
                    sinonimos.extend(syn.lemma_names())
                    for h in syn.hypernyms():
                        lwordnet.extend(h.lemma_names())
            except:
                pass

            diccionario_termino = {
                'termino': termino,
                'sinonimos': list(set(sinonimos)),
                'resource': info_termino.get('URI', ''),
                'dbpedia': info_termino.get('tipos', []),
                'wordnet': lwordnet
            }

            listado_tripletas.append(diccionario_termino)
            visitados.add(termino)
            terms_found += 1

    print(f"    Found {terms_found} relevant terms")

    if len(listado_tripletas) < 2:
        print(f"    ⚠️ Not enough terms for Topic {topic_id}, skipping")
        return None, None, None

    # Expand vocabulary with similarities (following original logic)
    df = pd.DataFrame(listado_tripletas)
    vocab_aux = []

    for _, row in df.iterrows():
        termino = row['termino']
        tipos = []

        dbpedia_tipos = row['dbpedia']
        if isinstance(dbpedia_tipos, list):
            tipos.extend(dbpedia_tipos)
        elif isinstance(dbpedia_tipos, str):
            tipos.extend(dbpedia_tipos.split(','))

        wordnet_tipos = row['wordnet']
        if isinstance(wordnet_tipos, list):
            tipos.extend(wordnet_tipos)

        # Clean types
        tipos_clean = []
        for t in tipos:
            el = str(t).split('/')[-1].split('#')[-1]
            el = ''.join([c for c in el if not c.isdigit()])
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

            vocab_aux.append(termino)
            if isinstance(sel, str):
                vocab_aux.append(sel)
            elif isinstance(sel, list):
                vocab_aux.extend(sel)
        except:
            continue

    # Lemmatize vocabulary
    vocab = set()
    for doc in nlp.pipe(vocab_aux):
        lemmatized = " ".join([token.lemma_.lower() for token in doc])
        vocab.add(lemmatized)

    # AQUÍ ES DONDE SE DEFINE terms - MOVER EL PRINT AQUÍ
    terms = list(vocab)

    if len(terms) < 2:
        print(f"    ⚠️ Final vocabulary too small for Topic {topic_id}, skipping")
        return None, None, None

    print(f"    Final vocabulary: {len(terms)} terms")

    # Calculate similarity matrix - AHORA terms YA ESTÁ DEFINIDO
    print(f"    Calculating similarity matrix...")
    M = np.array([[nlp(t1).similarity(nlp(t2)) for t2 in terms] for t1 in terms])

    print(f"    Similarity matrix shape: {M.shape}")
    print(f"    Similarity range: [{M.min():.3f}, {M.max():.3f}]")

    return M, terms, dictdbp

def compare_clustering_methods(similarity_matrix: np.ndarray, max_k: int = 8) -> dict:
    """
    Compare Ward, K-Means, and Spectral clustering across k=2 to max_k

    Args:
        similarity_matrix: Term similarity matrix (n_terms x n_terms)
        max_k: Maximum number of clusters to evaluate

    Returns:
        Dictionary with comparison results
    """
    n_terms = similarity_matrix.shape[0]
    max_k = min(max_k, n_terms - 1)

    if max_k < 2:
        return None

    # Distance matrix for Ward
    distance_matrix = 1 - similarity_matrix

    results = {
        'ward': [],
        'kmeans': [],
        'spectral': []
    }

    k_values = range(2, max_k + 1)

    for k in k_values:
        # Method 1: Ward Hierarchical
        try:
            linkage_matrix = shc.linkage(distance_matrix, method='ward')
            labels_ward = shc.fcluster(linkage_matrix, k, criterion='maxclust') - 1

            sil_ward = silhouette_score(distance_matrix, labels_ward, metric='precomputed')
            db_ward = davies_bouldin_score(similarity_matrix, labels_ward)
            ch_ward = calinski_harabasz_score(similarity_matrix, labels_ward)

            results['ward'].append({
                'k': k,
                'silhouette': float(sil_ward),
                'davies_bouldin': float(db_ward),
                'calinski_harabasz': float(ch_ward)
            })
        except Exception as e:
            print(f"      Ward error at k={k}: {e}")

        # Method 2: K-Means
        try:
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=42, max_iter=300)
            labels_kmeans = kmeans.fit_predict(similarity_matrix)

            sil_kmeans = silhouette_score(similarity_matrix, labels_kmeans)
            db_kmeans = davies_bouldin_score(similarity_matrix, labels_kmeans)
            ch_kmeans = calinski_harabasz_score(similarity_matrix, labels_kmeans)

            results['kmeans'].append({
                'k': k,
                'silhouette': float(sil_kmeans),
                'davies_bouldin': float(db_kmeans),
                'calinski_harabasz': float(ch_kmeans)
            })
        except Exception as e:
            print(f"      K-Means error at k={k}: {e}")

        # Method 3: Spectral
        try:
            # Spectral needs similarity matrix (affinity)
            spectral = SpectralClustering(
                n_clusters=k,
                affinity='precomputed',
                assign_labels='kmeans',
                random_state=42
            )
            labels_spectral = spectral.fit_predict(similarity_matrix)

            sil_spectral = silhouette_score(similarity_matrix, labels_spectral)
            db_spectral = davies_bouldin_score(similarity_matrix, labels_spectral)
            ch_spectral = calinski_harabasz_score(similarity_matrix, labels_spectral)

            results['spectral'].append({
                'k': k,
                'silhouette': float(sil_spectral),
                'davies_bouldin': float(db_spectral),
                'calinski_harabasz': float(ch_spectral)
            })
        except Exception as e:
            print(f"      Spectral error at k={k}: {e}")

    return results


def find_optimal_k_per_method(comparison_results: dict) -> dict:
    """
    Find optimal k for each method based on silhouette score

    Returns:
        {'ward': {'k': X, 'silhouette': Y}, 'kmeans': {...}, 'spectral': {...}}
    """
    optimal = {}

    for method in ['ward', 'kmeans', 'spectral']:
        if method in comparison_results and comparison_results[method]:
            # Find k with maximum silhouette
            best = max(comparison_results[method], key=lambda x: x['silhouette'])
            optimal[method] = {
                'k': best['k'],
                'silhouette': best['silhouette'],
                'davies_bouldin': best['davies_bouldin'],
                'calinski_harabasz': best['calinski_harabasz']
            }

    return optimal


def process_single_corpus(repo_name: str, nlp, output_dir: str) -> list:
    """
    Process a single corpus: load data, extract matrices, compare clustering

    Returns:
        List of topic results
    """
    print(f"\n{'=' * 70}")
    print(f"PROCESSING CORPUS: {repo_name.upper()}")
    print(f"{'=' * 70}")

    # Load data
    print("\n[1/4] Loading data...")
    try:
        df = load_processed_dataframe(repo_name)
        top_terms = load_top_terms(repo_name)
        vocabulary_dict = create_vocabulary_dict(df)
    except Exception as e:
        print(f"  ❌ Error loading data: {e}")
        return []

    # Load triplets
    triples_path = f'../data/triples_raw/{repo_name}/dataset_triplet_{repo_name}_new_simplificado.csv'
    if not os.path.exists(triples_path):
        print(f"  ❌ Triplets file not found: {triples_path}")
        return []

    df_triplets = pd.read_csv(triples_path)
    print(f"  ✅ Loaded {len(df_triplets)} triplets")

    # Get available topics
    available_topics = list(set(list(vocabulary_dict.keys()) + list(top_terms.keys())))
    available_topics = sorted([t for t in available_topics if t is not None])

    print(f"\n[2/4] Found {len(available_topics)} topics: {available_topics}")

    # Process each topic
    print("\n[3/4] Extracting similarity matrices and comparing clustering...")

    topic_results = []

    for topic_id in available_topics:
        # Extract similarity matrix
        M, terms, dictdbp = extract_similarity_matrix_for_topic(
            topic_id,
            repo_name,
            df_triplets,
            vocabulary_dict,
            top_terms,
            nlp
        )

        if M is None or terms is None:  # AÑADIR verificación de terms
            continue

        # Compare clustering methods - PASAR terms y nlp
        print(f"    Comparing clustering methods (k=2 to {MAX_CLUSTERS})...")
        comparison = compare_clustering_methods(
            M,
            terms,  # NUEVO
            nlp,  # NUEVO
            max_k=MAX_CLUSTERS
        )

        # Compare clustering methods

        if comparison is None:
            continue

        # Find optimal k for each method
        optimal = find_optimal_k_per_method(comparison)

        # Store results
        topic_result = {
            'corpus': repo_name,
            'topic_id': topic_id,
            'n_terms': len(terms),
            'comparison_full': comparison,
            'optimal_per_method': optimal
        }

        topic_results.append(topic_result)

        print(f"    ✅ Topic {topic_id}: Ward={optimal.get('ward', {}).get('silhouette', 0):.3f}, "
              f"KMeans={optimal.get('kmeans', {}).get('silhouette', 0):.3f}, "
              f"Spectral={optimal.get('spectral', {}).get('silhouette', 0):.3f}")

    print(f"\n[4/4] Successfully processed {len(topic_results)}/{len(available_topics)} topics")

    # Save individual corpus results
    corpus_output_dir = os.path.join(output_dir, repo_name)
    os.makedirs(corpus_output_dir, exist_ok=True)

    results_file = os.path.join(corpus_output_dir, f'{repo_name}_clustering_comparison_full.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(topic_results, f, indent=2, ensure_ascii=False)

    print(f"  ✅ Saved detailed results: {results_file}")

    return topic_results


def generate_global_comparison_table(all_results: dict, output_dir: str):
    """
    Generate comparison tables for the paper

    Args:
        all_results: {'amazon': [...], 'bbc': [...], 'reuters': [...]}
        output_dir: Output directory
    """
    print(f"\n{'=' * 70}")
    print("GENERATING GLOBAL COMPARISON TABLES")
    print(f"{'=' * 70}")

    # Flatten results into DataFrame
    rows = []

    for corpus, topics in all_results.items():
        for topic_result in topics:
            topic_id = topic_result['topic_id']
            optimal = topic_result['optimal_per_method']

            # Add row for each method
            for method in ['ward', 'kmeans', 'spectral']:
                if method in optimal:
                    rows.append({
                        'Corpus': corpus.upper(),
                        'Topic': topic_id,
                        'Method': method.capitalize(),
                        'Optimal_k': optimal[method]['k'],
                        'Silhouette': optimal[method]['silhouette'],
                        'Davies_Bouldin': optimal[method]['davies_bouldin'],
                        'Calinski_Harabasz': optimal[method]['calinski_harabasz']
                    })

    df = pd.DataFrame(rows)

    if len(df) == 0:
        print("  ⚠️ No results to generate table")
        return

    # Save full table
    csv_file = os.path.join(output_dir, 'clustering_comparison_all_methods.csv')
    df.to_csv(csv_file, index=False)
    print(f"\n✅ Full comparison table saved: {csv_file}")

    # Generate summary by corpus and method
    print("\n" + "=" * 70)
    print("SUMMARY TABLE - AVERAGE METRICS BY CORPUS AND METHOD")
    print("=" * 70)

    summary = df.groupby(['Corpus', 'Method']).agg({
        'Silhouette': 'mean',
        'Davies_Bouldin': 'mean',
        'Calinski_Harabasz': 'mean'
    }).reset_index()

    # Print markdown table
    print("\n**Table for Paper (Markdown format):**\n")
    print("| Corpus | Method | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ |")
    print("|--------|--------|--------------|------------------|---------------------|")

    for corpus in df['Corpus'].unique():
        corpus_data = summary[summary['Corpus'] == corpus]

        # Find best method for this corpus
        best_idx = corpus_data['Silhouette'].idxmax()
        best_method = corpus_data.loc[best_idx, 'Method']

        for _, row in corpus_data.iterrows():
            selected = "✓" if row['Method'] == best_method else ""
            print(f"| {row['Corpus']} | {row['Method']} | "
                  f"{row['Silhouette']:.3f} | {row['Davies_Bouldin']:.3f} | "
                  f"{row['Calinski_Harabasz']:.1f} | {selected} |")

    # Overall averages
    print("|--------|--------|--------------|------------------|---------------------|")
    overall = df.groupby('Method').agg({
        'Silhouette': 'mean',
        'Davies_Bouldin': 'mean',
        'Calinski_Harabasz': 'mean'
    }).reset_index()

    best_overall_idx = overall['Silhouette'].idxmax()
    best_overall_method = overall.loc[best_overall_idx, 'Method']

    for _, row in overall.iterrows():
        selected = "✓" if row['Method'] == best_overall_method else ""
        print(f"| **AVG** | **{row['Method']}** | "
              f"**{row['Silhouette']:.3f}** | **{row['Davies_Bouldin']:.3f}** | "
              f"**{row['Calinski_Harabasz']:.1f}** | {selected} |")

    print("\n" + "=" * 70)

    # Save summary
    summary_file = os.path.join(output_dir, 'clustering_comparison_summary.csv')
    summary.to_csv(summary_file, index=False)
    print(f"✅ Summary table saved: {summary_file}")

    # Generate visualization
    generate_comparison_plot(df, output_dir)

    # Generate statistics
    generate_statistics_report(df, output_dir)


def generate_comparison_plot(df: pd.DataFrame, output_dir: str):
    """Generate comparison visualization"""

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    corpora = df['Corpus'].unique()
    methods = ['Ward', 'Kmeans', 'Spectral']
    colors = {'Ward': '#3498db', 'Kmeans': '#e74c3c', 'Spectral': '#2ecc71'}

    x = np.arange(len(corpora))
    width = 0.25

    # Plot 1: Silhouette
    ax1 = axes[0]
    for i, method in enumerate(methods):
        method_data = df[df['Method'] == method].groupby('Corpus')['Silhouette'].mean()
        values = [method_data.get(corpus, 0) for corpus in corpora]
        ax1.bar(x + i * width, values, width, label=method, color=colors[method], edgecolor='black')

    ax1.set_xlabel('Corpus', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Silhouette Score (↑)', fontsize=12, fontweight='bold')
    ax1.set_title('Clustering Quality Comparison', fontsize=13, fontweight='bold')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(corpora)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Davies-Bouldin
    ax2 = axes[1]
    for i, method in enumerate(methods):
        method_data = df[df['Method'] == method].groupby('Corpus')['Davies_Bouldin'].mean()
        values = [method_data.get(corpus, 0) for corpus in corpora]
        ax2.bar(x + i * width, values, width, label=method, color=colors[method], edgecolor='black')

    ax2.set_xlabel('Corpus', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Davies-Bouldin Index (↓)', fontsize=12, fontweight='bold')
    ax2.set_title('Cluster Compactness Comparison', fontsize=13, fontweight='bold')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(corpora)
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)

    # Plot 3: Calinski-Harabasz
    ax3 = axes[2]
    for i, method in enumerate(methods):
        method_data = df[df['Method'] == method].groupby('Corpus')['Calinski_Harabasz'].mean()
        values = [method_data.get(corpus, 0) for corpus in corpora]
        ax3.bar(x + i * width, values, width, label=method, color=colors[method], edgecolor='black')

    ax3.set_xlabel('Corpus', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Calinski-Harabasz Score (↑)', fontsize=12, fontweight='bold')
    ax3.set_title('Cluster Density Comparison', fontsize=13, fontweight='bold')
    ax3.set_xticks(x + width)
    ax3.set_xticklabels(corpora)
    ax3.legend(fontsize=10)
    ax3.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    plot_file = os.path.join(output_dir, 'clustering_comparison_plot.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Comparison plot saved: {plot_file}")


def generate_statistics_report(df: pd.DataFrame, output_dir: str):
    """Generate detailed statistics report"""

    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("CLUSTERING COMPARISON - STATISTICAL REPORT")
    report_lines.append("=" * 70)
    report_lines.append("")

    # Overall statistics
    report_lines.append("OVERALL STATISTICS (All Corpora Combined)")
    report_lines.append("-" * 70)

    overall = df.groupby('Method').agg({
        'Silhouette': ['mean', 'std', 'min', 'max'],
        'Davies_Bouldin': ['mean', 'std', 'min', 'max']
    }).round(4)

    for method in df['Method'].unique():
        report_lines.append(f"\n{method.upper()}:")
        method_data = overall.loc[method]
        report_lines.append(
            f"  Silhouette:      {method_data[('Silhouette', 'mean')]:.4f} ± {method_data[('Silhouette', 'std')]:.4f}")
        report_lines.append(
            f"                   Range: [{method_data[('Silhouette', 'min')]:.4f}, {method_data[('Silhouette', 'max')]:.4f}]")
        report_lines.append(
            f"  Davies-Bouldin:  {method_data[('Davies_Bouldin', 'mean')]:.4f} ± {method_data[('Davies_Bouldin', 'std')]:.4f}")
        report_lines.append(
            f"                   Range: [{method_data[('Davies_Bouldin', 'min')]:.4f}, {method_data[('Davies_Bouldin', 'max')]:.4f}]")

    # Best method overall
    best_method = df.groupby('Method')['Silhouette'].mean().idxmax()
    best_sil = df.groupby('Method')['Silhouette'].mean().max()
    report_lines.append(f"\n🏆 Overall Best Method: {best_method.upper()} (Avg Silhouette: {best_sil:.4f})")

    # Per-corpus statistics
    report_lines.append("\n" + "=" * 70)
    report_lines.append("PER-CORPUS STATISTICS")
    report_lines.append("=" * 70)

    for corpus in df['Corpus'].unique():
        report_lines.append(f"\n{corpus}:")
        report_lines.append("-" * 40)

        corpus_data = df[df['Corpus'] == corpus]

        for method in df['Method'].unique():
            method_data = corpus_data[corpus_data['Method'] == method]

            if len(method_data) > 0:
                avg_sil = method_data['Silhouette'].mean()
                avg_db = method_data['Davies_Bouldin'].mean()
                n_topics = len(method_data)

                report_lines.append(
                    f"  {method.capitalize():10s}: Silhouette={avg_sil:.4f}, DB={avg_db:.4f} ({n_topics} topics)")

            # Best for this corpus
        corpus_best = corpus_data.groupby('Method')['Silhouette'].mean()
        if len(corpus_best) > 0:
            best_method_corpus = corpus_best.idxmax()
            best_sil_corpus = corpus_best.max()
            report_lines.append(f"  → Best: {best_method_corpus.upper()} (Silhouette: {best_sil_corpus:.4f})")

        # Pairwise comparisons
    report_lines.append("\n" + "=" * 70)
    report_lines.append("PAIRWISE COMPARISONS")
    report_lines.append("=" * 70)

    methods = df['Method'].unique()
    for i, method1 in enumerate(methods):
        for method2 in methods[i + 1:]:
            sil1 = df[df['Method'] == method1]['Silhouette'].mean()
            sil2 = df[df['Method'] == method2]['Silhouette'].mean()
            diff = abs(sil1 - sil2)
            pct = (diff / min(sil1, sil2)) * 100

            winner = method1 if sil1 > sil2 else method2

            report_lines.append(f"\n{method1.upper()} vs {method2.upper()}:")
            report_lines.append(f"  Silhouette difference: {diff:.4f} ({pct:.1f}% improvement)")
            report_lines.append(f"  Winner: {winner.upper()}")

    # Win rate per corpus
    report_lines.append("\n" + "=" * 70)
    report_lines.append("WIN RATES BY CORPUS")
    report_lines.append("=" * 70)

    for corpus in df['Corpus'].unique():
        corpus_data = df[df['Corpus'] == corpus]

        # Count wins per topic
        wins = defaultdict(int)
        topics = corpus_data['Topic'].unique()

        for topic in topics:
            topic_data = corpus_data[corpus_data['Topic'] == topic]
            if len(topic_data) > 0:
                winner = topic_data.loc[topic_data['Silhouette'].idxmax(), 'Method']
                wins[winner] += 1

        report_lines.append(f"\n{corpus}:")
        total_topics = len(topics)
        for method, count in sorted(wins.items(), key=lambda x: -x[1]):
            pct = (count / total_topics) * 100
            report_lines.append(f"  {method.capitalize():10s}: {count}/{total_topics} ({pct:.1f}%)")

    # Save report
    report_file = os.path.join(output_dir, 'clustering_comparison_statistics.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    print(f"✅ Statistics report saved: {report_file}")

    # Print summary to console
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # Overall winner
    overall_avg = df.groupby('Method')['Silhouette'].mean()
    winner = overall_avg.idxmax()
    winner_score = overall_avg.max()
    second = overall_avg.nlargest(2).index[-1]
    second_score = overall_avg.nlargest(2).iloc[-1]
    improvement = ((winner_score - second_score) / second_score) * 100

    print(f"\n🏆 OVERALL WINNER: {winner.upper()}")
    print(f"   Average Silhouette: {winner_score:.4f}")
    print(f"   Improvement over {second.upper()}: {improvement:.1f}%")

    # Consistency check
    corpus_winners = []
    for corpus in df['Corpus'].unique():
        corpus_best = df[df['Corpus'] == corpus].groupby('Method')['Silhouette'].mean().idxmax()
        corpus_winners.append(corpus_best)

    if len(set(corpus_winners)) == 1:
        print(f"\n✓ Consistent winner across ALL corpora: {corpus_winners[0].upper()}")
    else:
        print(f"\n⚠ Winners vary by corpus:")
        for corpus, winner_c in zip(df['Corpus'].unique(), corpus_winners):
            print(f"   {corpus}: {winner_c.upper()}")

    print("\n" + "=" * 70)


def main():
    """Main execution function"""

    print("\n" + "=" * 70)
    print("CLUSTERING COMPARISON ANALYSIS - POST-HOC")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  - Corpora: {', '.join(REPOSITORIES)}")
    print(f"  - Max clusters per topic: {MAX_CLUSTERS}")
    print(f"  - Methods: Ward, K-Means, Spectral")
    print(f"  - SpaCy model: {SPACY_MODEL}")

    # Create output directory
    output_dir = '../results/clustering_comparison'
    os.makedirs(output_dir, exist_ok=True)
    print(f"  - Output directory: {output_dir}")

    # Load SpaCy model
    print(f"\nLoading SpaCy model '{SPACY_MODEL}'...")
    try:
        nlp = spacy.load(SPACY_MODEL)
        print("  ✅ SpaCy model loaded")
    except OSError:
        print(f"  ❌ Error: SpaCy model '{SPACY_MODEL}' not found")
        print(f"     Install with: python -m spacy download {SPACY_MODEL}")
        return

    # Download WordNet
    try:
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    except:
        pass

    # Process each corpus
    all_results = {}

    for repo in REPOSITORIES:
        try:
            results = process_single_corpus(repo, nlp, output_dir)
            all_results[repo] = results
        except Exception as e:
            print(f"\n❌ Error processing {repo}: {e}")
            import traceback
            traceback.print_exc()
            all_results[repo] = []

    # Generate global comparison
    if any(len(results) > 0 for results in all_results.values()):
        generate_global_comparison_table(all_results, output_dir)
    else:
        print("\n⚠️ No results to generate comparison table")

    print("\n" + "=" * 70)
    print("✅ CLUSTERING COMPARISON COMPLETED")
    print("=" * 70)
    print(f"\nResults saved in: {output_dir}")
    print("\nGenerated files:")
    print("  - clustering_comparison_all_methods.csv (full data)")
    print("  - clustering_comparison_summary.csv (averages)")
    print("  - clustering_comparison_statistics.txt (detailed report)")
    print("  - clustering_comparison_plot.png (visualization)")
    print("  - {corpus}/{corpus}_clustering_comparison_full.json (per corpus)")


if __name__ == "__main__":
    main()