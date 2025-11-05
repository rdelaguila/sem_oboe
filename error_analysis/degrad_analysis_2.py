#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V5.0 ALTERNATIVE METRICS - Robustness Analysis
===============================================
Addresses NaN problem by using:
1. Topic-document entropy (from df_topic.pkl) instead of vocabulary entropy
2. Cluster-level overlap within topics
3. Aggregated metrics from all_topics_summary.json
4. Additional topic characteristics

Responds to reviewer: "Discuss explanation degradation when topics overlap
or entity extraction is noisy"

Author: OBOE Framework Team
"""

import os
import json
import pandas as pd
import numpy as np
import pickle
from collections import Counter
from scipy import stats
from scipy.stats import spearmanr, pearsonr, shapiro, levene, kruskal, mannwhitneyu, f_oneway
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# JSON SERIALIZATION HELPERS
# ============================================================================

def convert_to_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, (np.bool_, np.bool8)):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


def save_json_safely(data, filepath):
    """Save data to JSON with numpy type conversion"""
    serializable_data = convert_to_serializable(data)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, indent=2, ensure_ascii=False)


# ============================================================================
# NEW METRIC 1: TOPIC-DOCUMENT ENTROPY
# ============================================================================

def extract_topic_probabilities(topic_dist_list, topic_id):
    """
    Extract probabilities for a specific topic from topic_dist column

    topic_dist format: [(0, 0.983), (1, 0.017), ...]

    Parameters:
    -----------
    topic_dist_list : list of tuples
        List of (topic_id, probability) tuples
    topic_id : int
        Topic to extract

    Returns:
    --------
    float : Probability for this topic (0.0 if not found)
    """
    if not isinstance(topic_dist_list, list):
        return 0.0

    for tid, prob in topic_dist_list:
        if tid == topic_id:
            return prob

    return 0.0


def compute_topic_document_entropy(df_topic, topic_id):
    """
    Compute entropy of topic distribution across documents

    High entropy = topic is spread across many documents (dispersed)
    Low entropy = topic is concentrated in few documents (focused)

    Parameters:
    -----------
    df_topic : DataFrame
        Document-topic probability distribution
        Must have 'topic_dist' column: [(topic_id, prob), ...]
    topic_id : int
        Topic ID to analyze

    Returns:
    --------
    dict : Entropy metrics
    """

    if 'topic_dist' not in df_topic.columns:
        return {
            'entropy': 0.0,
            'concentration': 0.0,
            'n_docs': 0,
            'interpretation': 'topic_dist column not found'
        }

    # Extract probabilities for this topic across all documents
    probs = []

    for topic_dist in df_topic['topic_dist']:
        prob = extract_topic_probabilities(topic_dist, topic_id)
        if prob > 1e-10:  # Filter near-zero
            probs.append(prob)

    probs = np.array(probs)

    if len(probs) == 0:
        return {
            'entropy': 0.0,
            'concentration': 0.0,
            'n_docs': 0,
            'interpretation': 'No documents with this topic'
        }

    # Normalize to ensure sum = 1
    probs = probs / probs.sum()

    # Shannon entropy
    entropy = -np.sum(probs * np.log2(probs + 1e-10))

    # Max possible entropy (uniform distribution)
    max_entropy = np.log2(len(probs))

    # Normalized entropy [0, 1]
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0

    # Concentration: Gini coefficient
    # Higher Gini = more concentrated (few docs have high prob)
    sorted_probs = np.sort(probs)
    n = len(sorted_probs)
    gini = (2 * np.sum((n - np.arange(1, n + 1) + 1) * sorted_probs)) / (n * np.sum(sorted_probs)) - (n + 1) / n

    # Also compute: % of probability mass in top 10% of documents
    sorted_probs_desc = np.sort(probs)[::-1]
    top_10pct = int(np.ceil(len(probs) * 0.1))
    concentration_top10 = np.sum(sorted_probs_desc[:top_10pct])

    return {
        'entropy': float(entropy),
        'normalized_entropy': float(norm_entropy),
        'max_entropy': float(max_entropy),
        'concentration_gini': float(gini),
        'concentration_top10pct': float(concentration_top10),
        'n_docs': int(len(probs)),
        'mean_prob': float(np.mean(probs)),
        'std_prob': float(np.std(probs)),
        'interpretation': (
            'Highly dispersed' if norm_entropy > 0.8 else
            'Moderately dispersed' if norm_entropy > 0.5 else
            'Focused'
        )
    }


# ============================================================================
# NEW METRIC 2: CLUSTER OVERLAP WITHIN TOPICS
# ============================================================================

def compute_cluster_overlap_in_topic(explanations_file):
    """
    Compute vocabulary overlap between clusters within a topic

    Parameters:
    -----------
    explanations_file : str
        Path to explanations.json

    Returns:
    --------
    dict : Overlap statistics
    """

    with open(explanations_file, 'r') as f:
        explanations = json.load(f)

    cluster_vocabs = {}

    # Extract vocabulary for each cluster
    for cluster_id, data in explanations.items():
        key_phrases = data.get('key_phrases', [])

        # Extract terms from key phrases
        terms = set()
        for phrase in key_phrases:
            # Split phrases into terms
            terms.update(phrase.lower().split())

        cluster_vocabs[int(cluster_id)] = terms

    if len(cluster_vocabs) < 2:
        return {
            'mean_overlap': 0.0,
            'max_overlap': 0.0,
            'n_pairs': 0,
            'interpretation': 'Insufficient clusters'
        }

    # Compute pairwise overlaps (Jaccard)
    overlaps = []
    cluster_ids = sorted(cluster_vocabs.keys())

    for i, cid1 in enumerate(cluster_ids):
        for cid2 in cluster_ids[i + 1:]:
            vocab1 = cluster_vocabs[cid1]
            vocab2 = cluster_vocabs[cid2]

            if len(vocab1) == 0 or len(vocab2) == 0:
                continue

            intersection = len(vocab1 & vocab2)
            union = len(vocab1 | vocab2)

            jaccard = intersection / union if union > 0 else 0
            overlaps.append(jaccard)

    if not overlaps:
        return {
            'mean_overlap': 0.0,
            'max_overlap': 0.0,
            'n_pairs': 0,
            'interpretation': 'No valid pairs'
        }

    mean_overlap = np.mean(overlaps)
    max_overlap = np.max(overlaps)
    std_overlap = np.std(overlaps)

    return {
        'mean_overlap': float(mean_overlap),
        'max_overlap': float(max_overlap),
        'std_overlap': float(std_overlap),
        'n_pairs': len(overlaps),
        'interpretation': (
            'High overlap' if mean_overlap > 0.5 else
            'Moderate overlap' if mean_overlap > 0.3 else
            'Low overlap'
        )
    }


# ============================================================================
# LOAD DATA FROM YOUR STRUCTURE
# ============================================================================

def load_corpus_v5(corpus_name, base_data_dir='data'):
    """
    Load corpus data with V5 metrics

    Loads:
    1. df_topic.pkl → Topic-document distributions
    2. all_topics_summary.json → Aggregated metrics
    3. Cluster-level data
    """

    print(f"\n{'=' * 70}")
    print(f"LOADING: {corpus_name.upper()} (V5)")
    print(f"{'=' * 70}\n")

    lda_dir = os.path.join(base_data_dir, 'lda_eval', corpus_name)
    explanations_dir = os.path.join(base_data_dir, 'explanations_eng', corpus_name)

    corpus_data = {
        'name': corpus_name,
        'lda_dir': lda_dir,
        'explanations_dir': explanations_dir
    }

    # ========================================================================
    # 1. Load df_topic for topic-document entropy
    # ========================================================================
    print("[1/4] Loading df_topic...")

    df_topic_path = os.path.join(lda_dir, 'df_topic.pkl')
    if os.path.exists(df_topic_path):
        with open(df_topic_path, 'rb') as f:
            df_topic = pickle.load(f)
        corpus_data['df_topic'] = df_topic
        print(
            f"✓ Loaded: {len(df_topic)} documents, {len([c for c in df_topic.columns if c.startswith('topic_')])} topics")
    else:
        print(f"⚠️ df_topic.pkl not found")
        corpus_data['df_topic'] = None

    # ========================================================================
    # 2. Load all_topics_summary.json
    # ========================================================================
    print("[2/4] Loading all_topics_summary...")

    summary_path = os.path.join(explanations_dir, 'all_topics_summary.json')
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            all_topics_summary = json.load(f)
        corpus_data['all_topics_summary'] = all_topics_summary

        topics_metrics = all_topics_summary.get('topics_metrics', [])
        print(f"✓ Loaded: {len(topics_metrics)} topics")

        # Preview
        for tm in topics_metrics[:3]:
            print(
                f"  Topic {tm['topic_id']}: coh={tm['avg_coherence']:.2f}, rel={tm['avg_relevance']:.2f}, {tm['num_clusters']} clusters")
    else:
        print(f"⚠️ all_topics_summary.json not found")
        corpus_data['all_topics_summary'] = None

    # ========================================================================
    # 3. Compute topic-document entropy
    # ========================================================================
    print("[3/4] Computing topic-document entropy...")

    topic_entropies = {}

    if corpus_data['df_topic'] is not None and corpus_data['all_topics_summary'] is not None:
        topics_metrics = corpus_data['all_topics_summary'].get('topics_metrics', [])

        for tm in topics_metrics:
            topic_id = tm['topic_id']
            entropy_metrics = compute_topic_document_entropy(corpus_data['df_topic'], topic_id)
            topic_entropies[topic_id] = entropy_metrics

            print(f"  Topic {topic_id}: H={entropy_metrics['entropy']:.3f}, "
                  f"Hn={entropy_metrics['normalized_entropy']:.3f}, "
                  f"Gini={entropy_metrics['concentration_gini']:.3f}, "
                  f"n_docs={entropy_metrics['n_docs']}")

    corpus_data['topic_entropies'] = topic_entropies

    # ========================================================================
    # 4. Compute cluster overlap per topic
    # ========================================================================
    print("[4/4] Computing cluster overlap...")

    cluster_overlaps = {}

    if corpus_data['all_topics_summary'] is not None:
        topics_metrics = corpus_data['all_topics_summary'].get('topics_metrics', [])

        for tm in topics_metrics:
            topic_id = tm['topic_id']
            topic_dir = tm['output_dir']

            explanations_file = os.path.join(topic_dir, 'explanations.json')

            if os.path.exists(explanations_file):
                overlap_metrics = compute_cluster_overlap_in_topic(explanations_file)
                cluster_overlaps[topic_id] = overlap_metrics

                print(f"  Topic {topic_id}: mean_overlap={overlap_metrics['mean_overlap']:.3f}")
            else:
                print(f"  Topic {topic_id}: explanations.json not found")

    corpus_data['cluster_overlaps'] = cluster_overlaps

    print(f"\n✓ Corpus loaded successfully\n")

    return corpus_data


# ============================================================================
# BUILD ANALYSIS DATAFRAME (V5)
# ============================================================================

def build_v5_analysis_dataframe(all_corpus_data):
    """
    Build dataframe for V5 analysis

    Each row = one topic (not cluster)
    Combines:
    - Topic-document entropy
    - Cluster overlap
    - Aggregated quality metrics
    """

    print("\n[Building V5 DataFrame] Topic-level analysis...")

    rows = []

    for corpus_name, corpus_data in all_corpus_data.items():
        if corpus_data.get('all_topics_summary') is None:
            continue

        topics_metrics = corpus_data['all_topics_summary'].get('topics_metrics', [])

        for tm in topics_metrics:
            topic_id = tm['topic_id']

            row = {
                'corpus': corpus_name,
                'topic_id': topic_id,

                # Quality metrics (aggregated)
                'avg_coherence': tm['avg_coherence'],
                'avg_relevance': tm['avg_relevance'],
                'avg_coverage': tm['avg_coverage'],
                'num_clusters': tm['num_clusters'],

                # Topic-document entropy (CORRECTED)
                'topic_entropy': corpus_data['topic_entropies'].get(topic_id, {}).get('entropy', np.nan),
                'topic_entropy_normalized': corpus_data['topic_entropies'].get(topic_id, {}).get('normalized_entropy',
                                                                                                 np.nan),
                'topic_concentration_gini': corpus_data['topic_entropies'].get(topic_id, {}).get('concentration_gini',
                                                                                                 np.nan),
                'topic_concentration_top10': corpus_data['topic_entropies'].get(topic_id, {}).get(
                    'concentration_top10pct', np.nan),
                'topic_n_docs': corpus_data['topic_entropies'].get(topic_id, {}).get('n_docs', 0),
                'topic_mean_prob': corpus_data['topic_entropies'].get(topic_id, {}).get('mean_prob', np.nan),

                # Cluster overlap
                'cluster_mean_overlap': corpus_data['cluster_overlaps'].get(topic_id, {}).get('mean_overlap', np.nan),
                'cluster_max_overlap': corpus_data['cluster_overlaps'].get(topic_id, {}).get('max_overlap', np.nan),
                'cluster_n_pairs': corpus_data['cluster_overlaps'].get(topic_id, {}).get('n_pairs', 0),

                # Composite score
                'quality_score': (tm['avg_coherence'] + tm['avg_relevance'] + tm['avg_coverage']) / 3.0
            }

            rows.append(row)

    df = pd.DataFrame(rows)

    print(f"✓ Built dataframe: {len(df)} topics from {df['corpus'].nunique()} corpora")
    print(f"  Columns: {list(df.columns)}")

    # Show sample
    print("\nSample data:")
    print(df.head())

    return df


# ============================================================================
# STATISTICAL ANALYSIS (adapted from V4)
# ============================================================================

def correlation_analysis_v5(df):
    """V5 correlation analysis with new metrics"""

    print("\n[Correlations V5] Analyzing relationships...")

    independent_vars = [
        'topic_entropy',
        'topic_entropy_normalized',
        'topic_concentration_gini',
        'topic_concentration_top10',
        'cluster_mean_overlap',
        'num_clusters'
    ]

    dependent_vars = [
        'avg_coherence',
        'avg_relevance',
        'avg_coverage',
        'quality_score'
    ]

    results = {}

    for indep in independent_vars:
        for dep in dependent_vars:
            valid = df[[indep, dep]].dropna()

            if len(valid) < 5:
                print(f"  ⚠️ Skipping {indep} vs {dep}: only {len(valid)} samples")
                continue

            # Check variance
            if valid[indep].std() < 1e-10:
                print(f"  ⚠️ Skipping {indep} vs {dep}: no variance in {indep}")
                continue

            if valid[dep].std() < 1e-10:
                print(f"  ⚠️ Skipping {indep} vs {dep}: no variance in {dep}")
                continue

            # Compute correlations
            pearson_r, pearson_p = pearsonr(valid[indep], valid[dep])
            spearman_r, spearman_p = spearmanr(valid[indep], valid[dep])

            key = f"{indep}_vs_{dep}"

            results[key] = {
                'n_samples': len(valid),
                'parametric': {
                    'test': 'Pearson',
                    'r': float(pearson_r),
                    'p_value': float(pearson_p),
                    'significant': pearson_p < 0.05
                },
                'non_parametric': {
                    'test': 'Spearman',
                    'rho': float(spearman_r),
                    'p_value': float(spearman_p),
                    'significant': spearman_p < 0.05
                },
                'interpretation': (
                    'Strong negative' if spearman_r < -0.5 else
                    'Moderate negative' if spearman_r < -0.3 else
                    'Weak negative' if spearman_r < -0.1 else
                    'No correlation' if abs(spearman_r) <= 0.1 else
                    'Weak positive' if spearman_r < 0.3 else
                    'Moderate positive' if spearman_r < 0.5 else
                    'Strong positive'
                )
            }

            print(f"\n  {indep} vs {dep} (n={len(valid)}):")
            print(f"    Spearman: ρ={spearman_r:.3f}, p={spearman_p:.4f} "
                  f"{'***' if spearman_p < 0.001 else '**' if spearman_p < 0.01 else '*' if spearman_p < 0.05 else 'ns'}")
            print(f"    {results[key]['interpretation']}")

    return results


# ============================================================================
# MAIN PIPELINE V5
# ============================================================================

def run_v5_analysis(
        corpora=['bbc', 'reuters_activities', 'amazon'],
        base_data_dir='data',
        output_dir='results_v5_alternative'
):
    """
    V5 Main pipeline with alternative metrics
    """

    print("=" * 70)
    print("V5.0 ALTERNATIVE METRICS - Robustness Analysis")
    print("Using: Topic-document entropy + Cluster overlap + Aggregated metrics")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    # Load all corpora
    all_corpus_data = {}

    for corpus_name in corpora:
        try:
            corpus_data = load_corpus_v5(corpus_name, base_data_dir)
            all_corpus_data[corpus_name] = corpus_data
        except Exception as e:
            print(f"✗ Error loading {corpus_name}: {e}")
            continue

    if not all_corpus_data:
        print("✗ No corpora loaded successfully")
        return None

    # Build analysis dataframe
    df_v5 = build_v5_analysis_dataframe(all_corpus_data)

    # Save dataframe
    df_path = os.path.join(output_dir, 'v5_topic_analysis.csv')
    df_v5.to_csv(df_path, index=False)
    print(f"\n✓ Saved: {df_path}")

    # Descriptive statistics
    print("\n" + "=" * 70)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 70)
    print(df_v5.describe())

    # Correlation analysis
    correlations = correlation_analysis_v5(df_v5)

    # Save results
    results = {
        'n_topics': len(df_v5),
        'n_corpora': df_v5['corpus'].nunique(),
        'descriptive_stats': df_v5.describe().to_dict(),
        'correlations': correlations
    }

    results_path = os.path.join(output_dir, 'v5_robustness_analysis.json')
    save_json_safely(results, results_path)
    print(f"\n✓ Saved: {results_path}")

    # Generate summary
    generate_v5_summary(df_v5, correlations, output_dir)

    print("\n" + "=" * 70)
    print("✅ V5 Analysis complete!")
    print(f"📊 Results in: {output_dir}")
    print("=" * 70)

    return results


def generate_v5_summary(df, correlations, output_dir):
    """Generate human-readable summary"""

    summary_lines = []

    summary_lines.append("=" * 70)
    summary_lines.append("V5.0 ALTERNATIVE METRICS - SUMMARY")
    summary_lines.append("=" * 70)
    summary_lines.append("")

    summary_lines.append(f"Topics analyzed: {len(df)}")
    summary_lines.append(f"Corpora: {', '.join(df['corpus'].unique())}")
    summary_lines.append("")

    summary_lines.append("NEW METRICS:")
    summary_lines.append(
        f"  Topic-document entropy: M={df['topic_entropy'].mean():.3f} ± {df['topic_entropy'].std():.3f}")
    summary_lines.append(
        f"  Topic entropy (normalized): M={df['topic_entropy_normalized'].mean():.3f} ± {df['topic_entropy_normalized'].std():.3f}")
    summary_lines.append(
        f"  Topic concentration (Gini): M={df['topic_concentration_gini'].mean():.3f} ± {df['topic_concentration_gini'].std():.3f}")
    summary_lines.append(
        f"  Topic concentration (Top 10%): M={df['topic_concentration_top10'].mean():.3f} ± {df['topic_concentration_top10'].std():.3f}")
    summary_lines.append(
        f"  Cluster overlap: M={df['cluster_mean_overlap'].mean():.3f} ± {df['cluster_mean_overlap'].std():.3f}")
    summary_lines.append("")

    summary_lines.append("QUALITY METRICS:")
    summary_lines.append(f"  Coherence: M={df['avg_coherence'].mean():.2f} ± {df['avg_coherence'].std():.2f}")
    summary_lines.append(f"  Relevance: M={df['avg_relevance'].mean():.2f} ± {df['avg_relevance'].std():.2f}")
    summary_lines.append(f"  Coverage: M={df['avg_coverage'].mean():.2f} ± {df['avg_coverage'].std():.2f}")
    summary_lines.append("")

    summary_lines.append("=" * 70)
    summary_lines.append("KEY CORRELATIONS")
    summary_lines.append("=" * 70)

    for key, corr in correlations.items():
        if corr['non_parametric']['significant']:
            rho = corr['non_parametric']['rho']
            p = corr['non_parametric']['p_value']
            summary_lines.append(f"\n{key}:")
            summary_lines.append(f"  ρ = {rho:.3f}, p = {p:.4f} *")
            summary_lines.append(f"  {corr['interpretation']}")

    summary_lines.append("")
    summary_lines.append("=" * 70)

    summary_path = os.path.join(output_dir, 'V5_SUMMARY.txt')
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))

    print(f"\n✓ Summary saved: {summary_path}")
    print("\n" + '\n'.join(summary_lines))


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='V5.0 Alternative Metrics Analysis')
    parser.add_argument('--corpora', type=str, default='bbc,reuters_activities,amazon')
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--output_dir', type=str, default='results_v6_alternative')

    args = parser.parse_args()

    corpora = [c.strip() for c in args.corpora.split(',')]

    run_v5_analysis(
        corpora=corpora,
        base_data_dir=args.data_dir,
        output_dir=args.output_dir
    )
