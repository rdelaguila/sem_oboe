#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V4.0 TOPIC-CLUSTER ROBUSTNESS ANALYSIS - Production Ready
==========================================================
Automatically reads from your data structure:
  data/lda_eval/{corpus}/
  data/explanations_eng/{corpus}/{topic}/

Processes: bbc, reuters_activities, amazon

Author: OBOE Framework Team
"""

import os
import json
import pandas as pd
import numpy as np
import pickle
from collections import defaultdict, Counter
from scipy import stats
from scipy.stats import spearmanr, pearsonr, shapiro, levene, kruskal, mannwhitneyu, f_oneway
from pathlib import Path
import re
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
# DATA LOADING FROM YOUR STRUCTURE
# ============================================================================

def load_corpus_data(corpus_name, base_data_dir='data'):
    """
    Load all necessary data for a corpus from your directory structure

    Reads from:
    - data/lda_eval/{corpus}/
    - data/explanations_eng/{corpus}/

    Parameters:
    -----------
    corpus_name : str
        One of: 'bbc', 'reuters_activities', 'amazon'
    base_data_dir : str
        Base directory (default: 'data')

    Returns:
    --------
    dict : All loaded data
    """

    print(f"\n{'=' * 70}")
    print(f"LOADING CORPUS: {corpus_name.upper()}")
    print(f"{'=' * 70}")

    lda_dir = os.path.join(base_data_dir, 'lda_eval', corpus_name)
    explanations_dir = os.path.join(base_data_dir, 'explanations_eng', corpus_name)

    corpus_data = {
        'name': corpus_name,
        'lda_dir': lda_dir,
        'explanations_dir': explanations_dir
    }

    # ========================================================================
    # 1. Load LDA Model
    # ========================================================================
    print(f"\n[1/5] Loading LDA model...")

    lda_model_path = os.path.join(lda_dir, 'lda_model.pkl')
    if os.path.exists(lda_model_path):
        with open(lda_model_path, 'rb') as f:
            corpus_data['lda_model'] = pickle.load(f)
        print(f"✓ Loaded LDA model from: {lda_model_path}")
    else:
        print(f"⚠️ LDA model not found: {lda_model_path}")
        corpus_data['lda_model'] = None

    # ========================================================================
    # 2. Load Top Terms by Topic
    # ========================================================================
    print(f"\n[2/5] Loading topic vocabularies...")

    top_terms_path = os.path.join(lda_dir, 'top_terms_by_topic.pkl')
    if os.path.exists(top_terms_path):
        with open(top_terms_path, 'rb') as f:
            top_terms = pickle.load(f)

        # Convert to dict {topic_id: [term1, term2, ...]}
        topic_vocabularies = {}

        if isinstance(top_terms, dict):
            topic_vocabularies = top_terms
        elif isinstance(top_terms, list):
            # If it's a list of lists
            topic_vocabularies = {i: terms for i, terms in enumerate(top_terms)}

        corpus_data['topic_vocabularies'] = topic_vocabularies
        print(f"✓ Loaded vocabularies for {len(topic_vocabularies)} topics")

        # Print sample
        for topic_id in list(topic_vocabularies.keys())[:3]:
            vocab = topic_vocabularies[topic_id]
            print(f"  Topic {topic_id}: {len(vocab)} terms (sample: {vocab[:5]})")

    else:
        print(f"⚠️ Top terms not found: {top_terms_path}")
        print(f"  Attempting to extract from LDA model...")

        # Try to extract from LDA model
        if corpus_data['lda_model']:
            topic_vocabularies = extract_vocabularies_from_lda(corpus_data['lda_model'])
            corpus_data['topic_vocabularies'] = topic_vocabularies
            print(f"✓ Extracted vocabularies for {len(topic_vocabularies)} topics")
        else:
            corpus_data['topic_vocabularies'] = {}

    # ========================================================================
    # 3. Load df_topic (for entity frequencies)
    # ========================================================================
    print(f"\n[3/5] Loading document-topic distribution...")

    df_topic_path = os.path.join(lda_dir, 'df_topic.pkl')
    if os.path.exists(df_topic_path):
        with open(df_topic_path, 'rb') as f:
            df_topic = pickle.load(f)
        corpus_data['df_topic'] = df_topic
        print(f"✓ Loaded df_topic: {len(df_topic)} documents")
    else:
        print(f"⚠️ df_topic not found: {df_topic_path}")
        corpus_data['df_topic'] = None

    # ========================================================================
    # 4. Compute Entity Frequencies
    # ========================================================================
    print(f"\n[4/5] Computing entity frequencies...")

    entity_frequencies = Counter()

    if corpus_data['topic_vocabularies']:
        # Count from all topic vocabularies
        for topic_id, vocab in corpus_data['topic_vocabularies'].items():
            for term in vocab:
                entity_frequencies[str(term).lower().strip()] += 1

        print(f"✓ Computed frequencies for {len(entity_frequencies)} unique entities")
        print(f"  Total occurrences: {sum(entity_frequencies.values())}")

        # Statistics
        singleton_count = sum(1 for count in entity_frequencies.values() if count == 1)
        print(f"  Singleton entities: {singleton_count} ({100 * singleton_count / len(entity_frequencies):.1f}%)")

    corpus_data['entity_frequencies'] = entity_frequencies

    # ========================================================================
    # 5. Find Topic Evaluation Directories
    # ========================================================================
    print(f"\n[5/5] Finding topic evaluation directories...")

    topic_eval_dirs = {}

    if os.path.exists(explanations_dir):
        # List all topic directories
        for item in os.listdir(explanations_dir):
            topic_path = os.path.join(explanations_dir, item)

            if os.path.isdir(topic_path):
                # Check if it's a topic directory (topic_X or just X)
                if item.startswith('topic_'):
                    topic_id = int(item.split('_')[1])
                elif item.isdigit():
                    topic_id = int(item)
                else:
                    continue

                # Verify it has the required files
                explanations_file = os.path.join(topic_path, 'explanations.json')
                evaluations_file = os.path.join(topic_path, 'evaluations.json')

                if os.path.exists(explanations_file) and os.path.exists(evaluations_file):
                    topic_eval_dirs[topic_id] = topic_path
                    print(f"  ✓ Found topic {topic_id}: {topic_path}")

        print(f"\n✓ Found {len(topic_eval_dirs)} topics with evaluations")
    else:
        print(f"⚠️ Explanations directory not found: {explanations_dir}")

    corpus_data['topic_eval_dirs'] = topic_eval_dirs

    # ========================================================================
    # Summary
    # ========================================================================
    print(f"\n{'=' * 70}")
    print(f"CORPUS DATA LOADED: {corpus_name.upper()}")
    print(f"{'=' * 70}")
    print(f"  LDA Model: {'✓' if corpus_data['lda_model'] else '✗'}")
    print(f"  Topic Vocabularies: {len(corpus_data.get('topic_vocabularies', {}))}")
    print(f"  Entity Frequencies: {len(corpus_data.get('entity_frequencies', {}))}")
    print(f"  Topics with Evaluations: {len(corpus_data.get('topic_eval_dirs', {}))}")
    print(f"{'=' * 70}\n")

    return corpus_data


def extract_vocabularies_from_lda(lda_model, n_terms=30):
    """
    Extract top terms per topic from LDA model

    Supports both sklearn and gensim LDA models
    """

    topic_vocabularies = {}

    try:
        # Try sklearn format
        if hasattr(lda_model, 'components_'):
            feature_names = lda_model.get_feature_names_out()

            for topic_id, topic in enumerate(lda_model.components_):
                top_indices = topic.argsort()[-n_terms:][::-1]
                terms = [feature_names[i] for i in top_indices]
                topic_vocabularies[topic_id] = terms

        # Try gensim format
        elif hasattr(lda_model, 'show_topics'):
            for topic_id in range(lda_model.num_topics):
                terms = [word for word, _ in lda_model.show_topic(topic_id, n_terms)]
                topic_vocabularies[topic_id] = terms

        else:
            print("  ⚠️ Unknown LDA model format")

    except Exception as e:
        print(f"  ⚠️ Error extracting vocabularies: {e}")

    return topic_vocabularies


# ============================================================================
# TOPIC-LEVEL ANALYSIS (from previous script)
# ============================================================================

def compute_topic_vocabulary_entropy(topic_vocab, term_frequencies=None):
    """Compute Shannon entropy of topic vocabulary"""
    if not topic_vocab:
        return 0.0

    n_terms = len(topic_vocab)

    if term_frequencies:
        total = sum(term_frequencies.get(term, 1.0) for term in topic_vocab)
        probs = [term_frequencies.get(term, 1.0) / total for term in topic_vocab]
    else:
        probs = [1.0 / n_terms] * n_terms

    entropy = -sum(p * np.log2(p) for p in probs if p > 0)

    return entropy


def compute_vocabulary_overlap(vocab1, vocab2):
    """Compute Jaccard overlap between vocabularies"""
    set1 = set(vocab1)
    set2 = set(vocab2)

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return intersection / union if union > 0 else 0.0


def analyze_topic_vocabulary_overlap(topic_vocabs):
    """Analyze vocabulary overlap between all topic pairs"""
    print("\n[Topic Overlap] Analyzing vocabulary overlap...")

    topic_ids = sorted(topic_vocabs.keys())
    n_topics = len(topic_ids)

    overlaps = []
    overlap_matrix = np.zeros((n_topics, n_topics))

    for i, tid1 in enumerate(topic_ids):
        for j, tid2 in enumerate(topic_ids):
            if i < j:
                overlap = compute_vocabulary_overlap(topic_vocabs[tid1], topic_vocabs[tid2])
                overlaps.append(overlap)
                overlap_matrix[i, j] = overlap
                overlap_matrix[j, i] = overlap

    mean_overlap = np.mean(overlaps) if overlaps else 0
    std_overlap = np.std(overlaps) if overlaps else 0
    max_overlap = np.max(overlaps) if overlaps else 0

    # High-overlap pairs
    high_overlap_pairs = []
    threshold = mean_overlap + std_overlap if std_overlap > 0 else mean_overlap

    for i, tid1 in enumerate(topic_ids):
        for j, tid2 in enumerate(topic_ids):
            if i < j and overlap_matrix[i, j] > threshold:
                high_overlap_pairs.append({
                    'topic1': int(tid1),
                    'topic2': int(tid2),
                    'overlap': float(overlap_matrix[i, j])
                })

    results = {
        'n_topics': n_topics,
        'mean_overlap': float(mean_overlap),
        'std_overlap': float(std_overlap),
        'max_overlap': float(max_overlap),
        'min_overlap': float(np.min(overlaps)) if overlaps else 0,
        'high_overlap_pairs': high_overlap_pairs,
        'overlap_matrix': overlap_matrix.tolist()
    }

    print(f"✓ Mean overlap: {mean_overlap:.3f} ± {std_overlap:.3f}")
    print(f"  High-overlap pairs: {len(high_overlap_pairs)}")

    return results


def quantify_topic_entity_noise(topic_vocab, entity_frequencies):
    """Quantify noise in topic vocabulary"""

    if not topic_vocab:
        return {
            'noise_ratio': 0.0,
            'signal_to_noise_ratio': 0.0,
            'interpretation': 'No data'
        }

    # Count noise indicators
    noise_entities = set()
    for term in topic_vocab:
        if (entity_frequencies.get(term, 0) == 1 or
                len(term) < 3 or
                term.replace('.', '').replace(',', '').isdigit()):
            noise_entities.add(term)

    total_terms = len(topic_vocab)
    noise_count = len(noise_entities)
    signal_count = total_terms - noise_count

    noise_ratio = noise_count / total_terms if total_terms > 0 else 0.0
    snr = signal_count / noise_count if noise_count > 0 else float('inf')

    return {
        'total_terms': total_terms,
        'noise_count': noise_count,
        'signal_count': signal_count,
        'noise_ratio': float(noise_ratio),
        'signal_to_noise_ratio': float(snr),
        'interpretation': (
            'High noise' if noise_ratio > 0.5 else
            'Moderate noise' if noise_ratio > 0.3 else
            'Low noise'
        )
    }


# ============================================================================
# CLUSTER-LEVEL QUALITY (from previous script)
# ============================================================================

def load_cluster_evaluations(topic_dir):
    """Load QualIIT evaluations for clusters within a topic"""

    exp_path = os.path.join(topic_dir, 'explanations.json')
    if not os.path.exists(exp_path):
        return []

    with open(exp_path, 'r') as f:
        explanations = json.load(f)

    eval_path = os.path.join(topic_dir, 'evaluations.json')
    if os.path.exists(eval_path):
        with open(eval_path, 'r') as f:
            evaluations = json.load(f)
    else:
        evaluations = {}

    detail_path = os.path.join(topic_dir, 'detailed_analysis.json')
    if os.path.exists(detail_path):
        with open(detail_path, 'r') as f:
            detailed = json.load(f)
    else:
        detailed = {}

    clusters = []

    for cluster_id in explanations.keys():
        cluster_data = {
            'cluster_id': int(cluster_id),
            'explanation': explanations[cluster_id].get('explanation', ''),
            'coherence': evaluations.get(cluster_id, {}).get('coherence',
                                                             explanations[cluster_id].get('coherence', 0)),
            'relevance': evaluations.get(cluster_id, {}).get('relevance',
                                                             explanations[cluster_id].get('relevance', 0)),
            'coverage': evaluations.get(cluster_id, {}).get('coverage',
                                                            explanations[cluster_id].get('coverage', 0)),
            'justification': evaluations.get(cluster_id, {}).get('justification', ''),
            'strengths': evaluations.get(cluster_id, {}).get('strengths', []),
            'weaknesses': evaluations.get(cluster_id, {}).get('weaknesses', []),
            'key_phrases': explanations[cluster_id].get('key_phrases', []),
            'verification_passed': detailed.get(cluster_id, {}).get('verification_passed', False)
        }

        clusters.append(cluster_data)

    return clusters


# ============================================================================
# STATISTICAL ANALYSIS FUNCTIONS
# ============================================================================

def build_analysis_dataframe(topic_characteristics, all_cluster_evaluations):
    """Build dataframe for correlation analysis"""

    rows = []

    for topic_id, clusters in all_cluster_evaluations.items():
        topic_chars = topic_characteristics.get(topic_id, {})

        for cluster in clusters:
            row = {
                'topic_id': topic_id,
                'cluster_id': cluster['cluster_id'],
                'topic_vocab_entropy': topic_chars.get('vocab_entropy', 0),
                'topic_noise_ratio': topic_chars.get('noise_ratio', 0),
                'topic_snr': topic_chars.get('snr', 0),
                'topic_vocab_size': topic_chars.get('vocab_size', 0),
                'topic_avg_overlap': topic_chars.get('avg_overlap', 0),
                'topic_max_overlap': topic_chars.get('max_overlap', 0),
                'coherence': cluster['coherence'],
                'relevance': cluster['relevance'],
                'coverage': cluster['coverage'],
                'quality_score': (cluster['coherence'] + cluster['relevance'] + cluster['coverage']) / 3.0,
                'verification_passed': cluster['verification_passed']
            }
            rows.append(row)

    return pd.DataFrame(rows)


def test_assumptions(df, group_col='topic_id', value_cols=['coherence', 'relevance', 'coverage']):
    """Test statistical assumptions"""

    print("\n[Assumptions] Testing statistical assumptions...")

    results = {}

    # Normality
    for col in value_cols:
        data = df[col].dropna()
        if len(data) >= 3:
            stat, p = shapiro(data)
            results[f'{col}_normality'] = {
                'test': 'Shapiro-Wilk',
                'statistic': float(stat),
                'p_value': float(p),
                'is_normal': p > 0.05,
                'interpretation': 'Normal' if p > 0.05 else 'Non-normal'
            }
            print(f"  {col}: W={stat:.4f}, p={p:.4f} ({'Normal ✓' if p > 0.05 else 'Non-normal ⚠️'})")

    # Homogeneity of variance
    if group_col in df.columns and df[group_col].nunique() >= 2:
        for col in value_cols:
            groups = df.groupby(group_col)[col].apply(list)

            if len(groups) >= 2:
                stat, p = levene(*groups.values)
                results[f'{col}_homogeneity'] = {
                    'test': 'Levene',
                    'statistic': float(stat),
                    'p_value': float(p),
                    'homogeneous': p > 0.05,
                    'interpretation': 'Homogeneous' if p > 0.05 else 'Heterogeneous'
                }
                print(f"  {col} variance: W={stat:.4f}, p={p:.4f}")

    all_normal = all(r.get('is_normal', False) for k, r in results.items() if 'normality' in k)

    if all_normal:
        recommendation = "✓ Parametric tests appropriate"
    else:
        recommendation = "⚠️ Use non-parametric tests + report both"

    results['recommendation'] = recommendation
    print(f"\n  {recommendation}")

    return results


def correlation_analysis(df, independent_vars, dependent_vars):
    """Correlation analysis"""

    print("\n[Correlations] Topic characteristics → Cluster quality...")

    results = {}

    for indep in independent_vars:
        for dep in dependent_vars:
            valid = df[[indep, dep]].dropna()

            if len(valid) < 10:
                continue

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
                    'Moderate negative' if spearman_r < -0.3 else
                    'Weak negative' if spearman_r < -0.1 else
                    'No correlation' if abs(spearman_r) <= 0.1 else
                    'Weak positive' if spearman_r < 0.3 else
                    'Moderate positive'
                )
            }

            print(f"\n  {indep} vs {dep}:")
            print(f"    Spearman: ρ={spearman_r:.3f}, p={spearman_p:.4f} "
                  f"{'***' if spearman_p < 0.001 else '**' if spearman_p < 0.01 else '*' if spearman_p < 0.05 else 'ns'}")

    return results


def group_comparison_analysis(df, group_col='entropy_category', value_cols=['coherence', 'relevance', 'coverage']):
    """Group comparison analysis"""

    print("\n[Group Comparison] Comparing cluster quality across entropy categories...")

    # Check variance in entropy
    entropy_std = df['topic_vocab_entropy'].std()
    entropy_range = df['topic_vocab_entropy'].max() - df['topic_vocab_entropy'].min()

    # Always create entropy_category for post-hoc, even if we skip group comparison
    entropy_q1 = df['topic_vocab_entropy'].quantile(0.33)
    entropy_q3 = df['topic_vocab_entropy'].quantile(0.67)
    entropy_median = df['topic_vocab_entropy'].median()

    if entropy_std < 0.01 or entropy_range < 0.01:
        print(f"  ⚠️ Insufficient entropy variance (std={entropy_std:.4f}, range={entropy_range:.4f})")
        print(f"  Creating entropy_category for post-hoc but skipping group comparison")

        # Still create categories for post-hoc (use median split)
        df[group_col] = pd.cut(df['topic_vocab_entropy'],
                               bins=[0, entropy_median, float('inf')],
                               labels=['Low', 'High'],
                               duplicates='drop')

        return {
            'skipped': True,
            'reason': f'Insufficient variance (std={entropy_std:.4f})'
        }

    # Check for duplicate bins
    if abs(entropy_q1 - entropy_q3) < 0.001:
        print(f"  ⚠️ Q1 and Q3 are too close ({entropy_q1:.3f} vs {entropy_q3:.3f})")
        print(f"  Using median split instead of tertiles")

        # Use median split instead
        df[group_col] = pd.cut(df['topic_vocab_entropy'],
                               bins=[0, entropy_median, float('inf')],
                               labels=['Low', 'High'],
                               duplicates='drop')
    else:
        df[group_col] = pd.cut(df['topic_vocab_entropy'],
                               bins=[0, entropy_q1, entropy_q3, float('inf')],
                               labels=['Low', 'Medium', 'High'],
                               duplicates='drop')

    results = {}

    for col in value_cols:
        groups = df.groupby(group_col)[col].apply(list)

        if len(groups) < 2:
            print(f"  ⚠️ Insufficient groups for {col} ({len(groups)} groups)")
            continue

        # Filter out empty groups
        groups = {k: v for k, v in groups.items() if len(v) > 0}

        if len(groups) < 2:
            print(f"  ⚠️ Insufficient non-empty groups for {col}")
            continue

        f_stat, f_p = f_oneway(*groups.values)
        h_stat, h_p = kruskal(*groups.values)

        grand_mean = df[col].mean()
        ss_between = sum(len(group) * (np.mean(group) - grand_mean) ** 2 for group in groups.values)
        ss_total = sum((x - grand_mean) ** 2 for group in groups.values for x in group)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0

        results[col] = {
            'n_groups': len(groups),
            'group_names': list(groups.keys()),
            'group_means': {str(k): float(np.mean(v)) for k, v in groups.items()},
            'group_stds': {str(k): float(np.std(v)) for k, v in groups.items()},
            'parametric': {
                'test': 'ANOVA',
                'f_statistic': float(f_stat),
                'p_value': float(f_p),
                'significant': f_p < 0.05,
                'eta_squared': float(eta_squared),
                'effect_size': 'Large' if eta_squared > 0.14 else 'Medium' if eta_squared > 0.06 else 'Small'
            },
            'non_parametric': {
                'test': 'Kruskal-Wallis',
                'h_statistic': float(h_stat),
                'p_value': float(h_p),
                'significant': h_p < 0.05
            }
        }

        print(f"\n  {col} ({len(groups)} groups):")
        print(f"    ANOVA: F={f_stat:.2f}, p={f_p:.4f}, η²={eta_squared:.3f}")
        print(f"    Kruskal-Wallis: H={h_stat:.2f}, p={h_p:.4f}")

        # Print group statistics
        for group_name in sorted(groups.keys()):
            values = groups[group_name]
            print(f"      {group_name}: M={np.mean(values):.2f} ± {np.std(values):.2f} (n={len(values)})")

    return results


def posthoc_pairwise_comparisons(df, group_col='entropy_category', value_col='coherence'):
    """Post-hoc pairwise comparisons"""

    print(f"\n[Post-Hoc] Pairwise comparisons for {value_col}...")

    # Check if group_col exists
    if group_col not in df.columns:
        print(f"  ⚠️ Group column '{group_col}' not found, skipping post-hoc")
        return []

    groups = df.groupby(group_col)[value_col].apply(list)

    # Remove empty groups
    groups = {k: v for k, v in groups.items() if len(v) > 0}
    group_names = list(groups.keys())

    if len(group_names) < 2:
        print(f"  ⚠️ Need at least 2 groups, found {len(group_names)}")
        return []

    n_comparisons = len(group_names) * (len(group_names) - 1) / 2
    alpha_corrected = 0.05 / n_comparisons

    print(f"  Bonferroni α = {alpha_corrected:.4f} ({int(n_comparisons)} comparisons)")

    results = []

    for i in range(len(group_names)):
        for j in range(i + 1, len(group_names)):
            g1_name = group_names[i]
            g2_name = group_names[j]

            g1_data = groups[g1_name]
            g2_data = groups[g2_name]

            if len(g1_data) == 0 or len(g2_data) == 0:
                continue

            try:
                u_stat, p_val = mannwhitneyu(g1_data, g2_data, alternative='two-sided')

                significant = p_val < alpha_corrected

                results.append({
                    'group1': str(g1_name),
                    'group2': str(g2_name),
                    'u_statistic': float(u_stat),
                    'p_value': float(p_val),
                    'significant': significant,
                    'mean1': float(np.mean(g1_data)),
                    'mean2': float(np.mean(g2_data))
                })

                print(f"    {g1_name} vs {g2_name}: U={u_stat:.2f}, p={p_val:.4f} {'***' if significant else 'ns'}")

            except Exception as e:
                print(f"    ⚠️ Error comparing {g1_name} vs {g2_name}: {e}")
                continue

    return results


def run_topic_cluster_robustness_analysis(
        topic_vocabularies,
        topic_evaluation_dirs,
        entity_frequencies=None,
        output_dir='results_v4/',
        verbose=True
):
    """Main analysis pipeline"""

    print(f"\n{'=' * 70}")
    print("TOPIC-CLUSTER ROBUSTNESS ANALYSIS")
    print(f"{'=' * 70}")

    os.makedirs(output_dir, exist_ok=True)

    # Phase 1: Topic characteristics
    print("\n[PHASE 1] Analyzing Topic Characteristics...")

    topic_characteristics = {}

    for topic_id, vocab in topic_vocabularies.items():
        entropy = compute_topic_vocabulary_entropy(vocab)

        topic_characteristics[topic_id] = {
            'vocab_size': len(vocab),
            'vocab_entropy': entropy
        }

    print(f"✓ Computed entropy for {len(topic_characteristics)} topics")

    # Overlap analysis
    overlap_analysis = analyze_topic_vocabulary_overlap(topic_vocabularies)

    overlap_matrix = np.array(overlap_analysis['overlap_matrix'])
    for i, topic_id in enumerate(sorted(topic_vocabularies.keys())):
        if len(overlap_matrix) > 0:
            avg_overlap = np.mean([overlap_matrix[i, j] for j in range(len(overlap_matrix)) if i != j])
            max_overlap = np.max([overlap_matrix[i, j] for j in range(len(overlap_matrix)) if i != j]) if len(
                overlap_matrix) > 1 else 0

            topic_characteristics[topic_id]['avg_overlap'] = float(avg_overlap)
            topic_characteristics[topic_id]['max_overlap'] = float(max_overlap)

    # Noise analysis
    if entity_frequencies:
        for topic_id, vocab in topic_vocabularies.items():
            noise_metrics = quantify_topic_entity_noise(vocab, entity_frequencies)

            topic_characteristics[topic_id]['noise_ratio'] = noise_metrics['noise_ratio']
            topic_characteristics[topic_id]['snr'] = noise_metrics['signal_to_noise_ratio']

    # Phase 2: Load cluster evaluations
    print("\n[PHASE 2] Loading Cluster Evaluations...")

    all_cluster_evaluations = {}

    for topic_id, eval_dir in topic_evaluation_dirs.items():
        clusters = load_cluster_evaluations(eval_dir)
        all_cluster_evaluations[topic_id] = clusters

        if clusters:
            avg_coh = np.mean([c['coherence'] for c in clusters])
            print(f"  Topic {topic_id}: {len(clusters)} clusters, avg coherence={avg_coh:.2f}")

    total_clusters = sum(len(clusters) for clusters in all_cluster_evaluations.values())
    print(f"\n✓ Loaded {total_clusters} clusters from {len(all_cluster_evaluations)} topics")

    # Phase 3: Build dataframe
    df_analysis = build_analysis_dataframe(topic_characteristics, all_cluster_evaluations)

    df_path = os.path.join(output_dir, 'topic_cluster_analysis.csv')
    df_analysis.to_csv(df_path, index=False)
    print(f"\n✓ Saved: {df_path}")

    # Phase 4: Statistical analysis
    print("\n[PHASE 4] STATISTICAL ANALYSIS")
    print("=" * 70)

    comprehensive_results = {
        'topic_characteristics': {str(k): v for k, v in topic_characteristics.items()},
        'overlap_analysis': overlap_analysis,
        'n_topics': len(topic_vocabularies),
        'n_clusters': total_clusters
    }

    assumptions = test_assumptions(df_analysis)
    comprehensive_results['assumptions'] = assumptions

    independent_vars = ['topic_vocab_entropy', 'topic_noise_ratio', 'topic_avg_overlap']
    dependent_vars = ['coherence', 'relevance', 'coverage', 'quality_score']

    correlations = correlation_analysis(df_analysis, independent_vars, dependent_vars)
    comprehensive_results['correlations'] = correlations

    group_comparisons = group_comparison_analysis(df_analysis)
    comprehensive_results['group_comparisons'] = group_comparisons

    posthoc_coherence = posthoc_pairwise_comparisons(df_analysis, value_col='coherence')
    comprehensive_results['posthoc'] = {'coherence': posthoc_coherence}

    # Save results
    results_path = os.path.join(output_dir, 'comprehensive_robustness_analysis.json')
    save_json_safely(comprehensive_results, results_path)
    print(f"\n✓ Saved: {results_path}")

    # Summary
    print(f"\n{'=' * 70}")
    print("ANALYSIS SUMMARY")
    print(f"{'=' * 70}")
    print(f"Topics: {len(topic_vocabularies)}, Clusters: {total_clusters}")

    if 'topic_vocab_entropy_vs_coherence' in correlations:
        corr = correlations['topic_vocab_entropy_vs_coherence']
        print(
            f"\nEntropy → Coherence: ρ={corr['non_parametric']['rho']:.3f}, p={corr['non_parametric']['p_value']:.4f}")

    print(f"{'=' * 70}\n")

    return comprehensive_results


def generate_executive_summary(all_results, output_base_dir):
    """
    Generate executive summary report

    Creates a human-readable summary of all results
    """

    print("\n[Summary] Generating executive summary...")

    summary_lines = []

    summary_lines.append("=" * 70)
    summary_lines.append("V4.0 TOPIC-CLUSTER ROBUSTNESS ANALYSIS")
    summary_lines.append("EXECUTIVE SUMMARY")
    summary_lines.append("=" * 70)
    summary_lines.append("")

    # Overall statistics
    total_topics = sum(r.get('n_topics', 0) for r in all_results.values())
    total_clusters = sum(r.get('n_clusters', 0) for r in all_results.values())

    summary_lines.append(f"DATASETS ANALYZED: {len(all_results)}")
    summary_lines.append(f"Total Topics: {total_topics}")
    summary_lines.append(f"Total Clusters: {total_clusters}")
    summary_lines.append("")
    summary_lines.append("=" * 70)

    # Results by corpus
    for corpus_name, results in all_results.items():
        summary_lines.append("")
        summary_lines.append(f"CORPUS: {corpus_name.upper()}")
        summary_lines.append("-" * 70)

        summary_lines.append(f"  Topics: {results.get('n_topics', 0)}")
        summary_lines.append(f"  Clusters: {results.get('n_clusters', 0)}")

        # Overlap analysis
        if 'overlap_analysis' in results:
            overlap = results['overlap_analysis']
            summary_lines.append(f"  Mean topic overlap: {overlap.get('mean_overlap', 0):.3f}")

        summary_lines.append("")

        # Key correlations
        if 'correlations' in results:
            corrs = results['correlations']

            summary_lines.append("  KEY CORRELATIONS:")

            # Entropy vs Quality
            if 'topic_vocab_entropy_vs_coherence' in corrs:
                c = corrs['topic_vocab_entropy_vs_coherence']
                rho = c['non_parametric']['rho']
                p = c['non_parametric']['p_value']
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'

                summary_lines.append(f"    Entropy → Coherence:")
                summary_lines.append(f"      Spearman ρ = {rho:.3f}, p = {p:.4f} {sig}")
                summary_lines.append(f"      Interpretation: {c.get('interpretation', 'N/A')}")

            if 'topic_vocab_entropy_vs_relevance' in corrs:
                c = corrs['topic_vocab_entropy_vs_relevance']
                rho = c['non_parametric']['rho']
                p = c['non_parametric']['p_value']
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'

                summary_lines.append(f"    Entropy → Relevance:")
                summary_lines.append(f"      Spearman ρ = {rho:.3f}, p = {p:.4f} {sig}")

            # Noise vs Quality
            if 'topic_noise_ratio_vs_coherence' in corrs:
                c = corrs['topic_noise_ratio_vs_coherence']
                rho = c['non_parametric']['rho']
                p = c['non_parametric']['p_value']
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'

                summary_lines.append(f"    Noise → Coherence:")
                summary_lines.append(f"      Spearman ρ = {rho:.3f}, p = {p:.4f} {sig}")

        summary_lines.append("")

        # Group comparisons
        if 'group_comparisons' in results and results['group_comparisons']:
            gc = results['group_comparisons']

            if 'skipped' in gc:
                summary_lines.append("  GROUP COMPARISON:")
                summary_lines.append(f"    ⚠️ Skipped - {gc.get('reason', 'Unknown reason')}")
            elif 'coherence' in gc:
                summary_lines.append("  GROUP COMPARISON (Coherence):")

                coh = gc['coherence']

                # Group means
                if 'group_means' in coh:
                    summary_lines.append("    Group Means:")
                    for group_name, mean_val in sorted(coh['group_means'].items()):
                        summary_lines.append(f"      {group_name}: {mean_val:.2f}")

                # ANOVA
                if 'parametric' in coh:
                    param = coh['parametric']
                    f_stat = param.get('f_statistic', 0)
                    p_val = param.get('p_value', 1)
                    eta2 = param.get('eta_squared', 0)
                    effect = param.get('effect_size', 'Unknown')
                    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'

                    summary_lines.append(f"    ANOVA: F = {f_stat:.2f}, p = {p_val:.4f} {sig}")
                    summary_lines.append(f"    Effect size: η² = {eta2:.3f} ({effect})")

        summary_lines.append("")

    # Cross-corpus summary
    summary_lines.append("=" * 70)
    summary_lines.append("CROSS-CORPUS SUMMARY")
    summary_lines.append("=" * 70)
    summary_lines.append("")

    # Collect entropy effects
    entropy_effects = []
    for corpus_name, results in all_results.items():
        if 'correlations' in results:
            if 'topic_vocab_entropy_vs_coherence' in results['correlations']:
                c = results['correlations']['topic_vocab_entropy_vs_coherence']
                rho = c['non_parametric']['rho']
                p = c['non_parametric']['p_value']
                entropy_effects.append({
                    'corpus': corpus_name,
                    'rho': rho,
                    'p_value': p,
                    'significant': p < 0.05
                })

    if entropy_effects:
        summary_lines.append("ENTROPY EFFECT ACROSS CORPORA:")
        summary_lines.append("")

        for effect in entropy_effects:
            sig = '***' if effect['p_value'] < 0.001 else '**' if effect['p_value'] < 0.01 else '*' if effect[
                                                                                                           'p_value'] < 0.05 else 'ns'
            summary_lines.append(
                f"  {effect['corpus']:<20} ρ = {effect['rho']:>6.3f}, p = {effect['p_value']:.4f} {sig}")

        summary_lines.append("")

        # Mean effect
        rhos = [e['rho'] for e in entropy_effects]
        mean_rho = np.mean(rhos)
        std_rho = np.std(rhos)

        summary_lines.append(f"  Mean Effect: ρ = {mean_rho:.3f} ± {std_rho:.3f}")

        n_significant = sum(1 for e in entropy_effects if e['significant'])
        summary_lines.append(f"  Significant in: {n_significant}/{len(entropy_effects)} corpora")

        if all(r < 0 for r in rhos):
            summary_lines.append("  Direction: CONSISTENT NEGATIVE (✓)")
        elif all(r > 0 for r in rhos):
            summary_lines.append("  Direction: CONSISTENT POSITIVE")
        else:
            summary_lines.append("  Direction: INCONSISTENT")

    summary_lines.append("")
    summary_lines.append("=" * 70)
    summary_lines.append("KEY FINDINGS")
    summary_lines.append("=" * 70)
    summary_lines.append("")

    # Determine key findings
    if entropy_effects:
        mean_rho = np.mean([e['rho'] for e in entropy_effects])
        n_sig = sum(1 for e in entropy_effects if e['significant'])

        if n_sig >= 2 and mean_rho < -0.2:
            summary_lines.append("✓ STRONG EVIDENCE: Topic entropy negatively affects cluster quality")
            summary_lines.append(f"  - Consistent across {n_sig} corpora")
            summary_lines.append(f"  - Mean correlation: ρ = {mean_rho:.3f}")
            summary_lines.append("  - Interpretation: High-entropy topics → Lower quality clusters")
        elif n_sig >= 1:
            summary_lines.append("✓ MODERATE EVIDENCE: Topic entropy affects cluster quality")
            summary_lines.append(f"  - Significant in {n_sig} corpus")
        else:
            summary_lines.append("⚠️ LIMITED EVIDENCE: Entropy effect not consistently significant")

    summary_lines.append("")
    summary_lines.append("=" * 70)
    summary_lines.append("RECOMMENDATIONS FOR PAPER")
    summary_lines.append("=" * 70)
    summary_lines.append("")

    if entropy_effects and any(e['significant'] for e in entropy_effects):
        summary_lines.append("FOR MAIN TEXT:")
        summary_lines.append("  - Report entropy-quality correlation across corpora")
        summary_lines.append("  - Include cross-corpus table with correlations")
        summary_lines.append("  - Emphasize robustness (multiple datasets)")
        summary_lines.append("")
        summary_lines.append("FOR SUPPLEMENTARY:")
        summary_lines.append("  - Detailed statistics per corpus")
        summary_lines.append("  - Group comparison results")
        summary_lines.append("  - Post-hoc pairwise comparisons")
    else:
        summary_lines.append("RECOMMENDATION:")
        summary_lines.append("  - Report as exploratory analysis")
        summary_lines.append("  - Emphasize robustness of framework to entropy variation")
        summary_lines.append("  - Suggest further investigation with larger samples")

    summary_lines.append("")
    summary_lines.append("=" * 70)
    summary_lines.append(f"Analysis generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append("=" * 70)

    # Save summary
    summary_path = os.path.join(output_base_dir, 'EXECUTIVE_SUMMARY.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))

    print(f"✓ Executive summary saved to: {summary_path}")

    # Print to console
    print("\n" + '\n'.join(summary_lines))

    return summary_path


# ============================================================================
# MULTI-CORPUS RUNNER
# ============================================================================

def run_all_corpora(
        corpora=['bbc', 'reuters_activities', 'amazon'],
        base_data_dir='data',
        output_base_dir='results_v4_production'
):
    """
    Run V4.0 analysis for all specified corpora

    Parameters:
    -----------
    corpora : list
        List of corpus names to process
    base_data_dir : str
        Base data directory
    output_base_dir : str
        Base output directory

    Returns:
    --------
    dict : Results for all corpora
    """

    print("=" * 70)
    print("V4.0 TOPIC-CLUSTER ROBUSTNESS ANALYSIS")
    print("Multi-Corpus Production Run")
    print("=" * 70)
    print(f"\nCorpora to process: {', '.join(corpora)}")
    print(f"Base data directory: {base_data_dir}")
    print(f"Output directory: {output_base_dir}")
    print("=" * 70)

    os.makedirs(output_base_dir, exist_ok=True)

    all_results = {}

    for corpus_name in corpora:
        print(f"\n\n{'#' * 70}")
        print(f"# PROCESSING CORPUS: {corpus_name.upper()}")
        print(f"{'#' * 70}\n")

        try:
            # Load corpus data
            corpus_data = load_corpus_data(corpus_name, base_data_dir)

            # Check if we have necessary data
            if not corpus_data.get('topic_vocabularies'):
                print(f"✗ Skipping {corpus_name}: No topic vocabularies found")
                continue

            if not corpus_data.get('topic_eval_dirs'):
                print(f"✗ Skipping {corpus_name}: No evaluation directories found")
                continue

            # Run analysis
            output_dir = os.path.join(output_base_dir, corpus_name)
            os.makedirs(output_dir, exist_ok=True)

            results = run_topic_cluster_robustness_analysis(
                topic_vocabularies=corpus_data['topic_vocabularies'],
                topic_evaluation_dirs=corpus_data['topic_eval_dirs'],
                entity_frequencies=corpus_data.get('entity_frequencies'),
                output_dir=output_dir,
                verbose=True
            )

            all_results[corpus_name] = results

            print(f"\n✅ {corpus_name.upper()} analysis complete!")
            print(f"📊 Results saved to: {output_dir}")

        except Exception as e:
            print(f"\n✗ Error processing {corpus_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # ========================================================================
    # Cross-Corpus Comparison
    # ========================================================================
    if len(all_results) > 1:
        print(f"\n\n{'=' * 70}")
        print("CROSS-CORPUS COMPARISON")
        print(f"{'=' * 70}\n")

        comparison = compare_corpora(all_results)

        comparison_path = os.path.join(output_base_dir, 'cross_corpus_comparison.json')
        save_json_safely(comparison, comparison_path)

        print(f"\n✓ Cross-corpus comparison saved to: {comparison_path}")

    # ========================================================================
    # Final Summary
    # ========================================================================
    print(f"\n\n{'=' * 70}")
    print("FINAL SUMMARY")
    print(f"{'=' * 70}")
    print(f"\nProcessed corpora: {len(all_results)}/{len(corpora)}")

    for corpus_name, results in all_results.items():
        print(f"\n{corpus_name.upper()}:")
        print(f"  Topics: {results.get('n_topics', 0)}")
        print(f"  Clusters: {results.get('n_clusters', 0)}")

        if 'correlations' in results:
            if 'topic_vocab_entropy_vs_coherence' in results['correlations']:
                corr = results['correlations']['topic_vocab_entropy_vs_coherence']
                rho = corr['non_parametric']['rho']
                p = corr['non_parametric']['p_value']
                print(f"  Entropy ↔ Coherence: ρ={rho:.3f}, p={p:.4f}")

    print(f"\n{'=' * 70}")
    print(f"✅ All analyses complete!")
    print(f"📊 Results saved to: {output_base_dir}")

    # Generate executive summary
    summary_path = generate_executive_summary(all_results, output_base_dir)

    print(f"\n📋 Executive summary: {summary_path}")
    print(f"{'=' * 70}\n")

    return all_results


def compare_corpora(all_results):
    """Compare results across corpora"""

    comparison = {
        'corpora': {},
        'summary': {}
    }

    entropy_correlations = []
    noise_correlations = []

    for corpus_name, results in all_results.items():
        corpus_summary = {
            'n_topics': results.get('n_topics', 0),
            'n_clusters': results.get('n_clusters', 0)
        }

        if 'correlations' in results:
            corrs = results['correlations']

            if 'topic_vocab_entropy_vs_coherence' in corrs:
                rho = corrs['topic_vocab_entropy_vs_coherence']['non_parametric']['rho']
                p = corrs['topic_vocab_entropy_vs_coherence']['non_parametric']['p_value']
                corpus_summary['entropy_vs_coherence'] = {
                    'rho': rho,
                    'p_value': p,
                    'significant': p < 0.05
                }
                entropy_correlations.append(rho)

            if 'topic_noise_ratio_vs_coherence' in corrs:
                rho = corrs['topic_noise_ratio_vs_coherence']['non_parametric']['rho']
                noise_correlations.append(rho)

        comparison['corpora'][corpus_name] = corpus_summary

    # Summary statistics
    if entropy_correlations:
        comparison['summary']['entropy_effect'] = {
            'mean_rho': float(np.mean(entropy_correlations)),
            'std_rho': float(np.std(entropy_correlations)),
            'min_rho': float(np.min(entropy_correlations)),
            'max_rho': float(np.max(entropy_correlations)),
            'consistent_negative': all(r < -0.1 for r in entropy_correlations)
        }

    if noise_correlations:
        comparison['summary']['noise_effect'] = {
            'mean_rho': float(np.mean(noise_correlations)),
            'std_rho': float(np.std(noise_correlations))
        }

    # Print table
    print("\nCross-Corpus Summary Table:")
    print("-" * 70)
    print(f"{'Corpus':<20} {'Topics':<10} {'Clusters':<10} {'Entropy↔Coh':<15}")
    print("-" * 70)

    for corpus_name, data in comparison['corpora'].items():
        rho = data.get('entropy_vs_coherence', {}).get('rho', 0)
        print(f"{corpus_name:<20} {data['n_topics']:<10} {data['n_clusters']:<10} {rho:<15.3f}")

    print("-" * 70)

    if 'entropy_effect' in comparison['summary']:
        ent = comparison['summary']['entropy_effect']
        print(f"\nMean entropy effect: {ent['mean_rho']:.3f} ± {ent['std_rho']:.3f}")
        print(f"Consistent negative: {'Yes ✓' if ent['consistent_negative'] else 'No'}")

    return comparison


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='V4.0 Topic-Cluster Analysis - Production',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--corpora', type=str,
                        default='bbc,reuters_activities,amazon',
                        help='Comma-separated corpus names')
    parser.add_argument('--data_dir', type=str, default='../data',
                        help='Base data directory')
    parser.add_argument('--output_dir', type=str, default='degrad_analysis',
                        help='Output directory')

    args = parser.parse_args()

    corpora = [c.strip() for c in args.corpora.split(',')]

    run_all_corpora(
        corpora=corpora,
        base_data_dir=args.data_dir,
        output_base_dir=args.output_dir
    )