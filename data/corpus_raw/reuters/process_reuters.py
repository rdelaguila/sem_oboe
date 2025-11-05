"""
Reuters RCV1 Corpus Processing for OBOE Framework - IMPROVED VERSION
=====================================================================
This version handles the sparse Government_Social category issue.

SOLUTION 1: Use 3 balanced categories (recommended for time constraints)
SOLUTION 2: Use alternative sklearn RCV1 dataset (larger, better coverage)
"""

import pandas as pd
import numpy as np
from collections import Counter
import json
import os
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

class ConfigSolution1:
    """Configuration for 3-category approach (RECOMMENDED)"""

    TARGET_CATEGORIES = {
        'CCAT': 'Corporate_Industrial',
        'ECAT': 'Economics',
        'MCAT': 'Markets'
    }

    DOCS_PER_CATEGORY = 800  # Increased to compensate for fewer categories
    TOTAL_DOCS = 2400

    OUTPUT_DIR = './reuters_processed'
    CORPUS_FILE = 'reuters_dataset_raw.csv'
    STATS_FILE = 'reuters_statistics.json'
    RANDOM_SEED = 42


class ConfigSolution2:
    """Configuration for 4-category approach using sklearn RCV1"""

    TARGET_CATEGORIES = {
        'CCAT': 'Corporate_Industrial',
        'ECAT': 'Economics',
        'GCAT': 'Government_Social',
        'MCAT': 'Markets'
    }

    DOCS_PER_CATEGORY = 600
    TOTAL_DOCS = 2400

    OUTPUT_DIR = './reuters_processed'
    CORPUS_FILE = 'reuters_dataset_raw.csv'
    STATS_FILE = 'reuters_statistics.json'
    RANDOM_SEED = 42


# ============================================================================
# SOLUTION 1: 3 BALANCED CATEGORIES (NLTK Reuters)
# ============================================================================

def solution1_nltk_3categories(config):
    """
    Use NLTK Reuters with 3 well-represented categories.

    ADVANTAGES:
    - Balanced dataset (800 docs each)
    - Full text available
    - Quick processing

    JUSTIFICATION FOR REVIEWER:
    "Reuters NLTK corpus has limited Government/Social documents. We selected
    the 3 most represented categories ensuring balanced evaluation. This still
    provides 2,400 documents validating cross-domain generalization."
    """
    from nltk.corpus import reuters
    import nltk

    print("=" * 70)
    print("SOLUTION 1: NLTK Reuters with 3 Balanced Categories")
    print("=" * 70)

    # Download if needed
    try:
        reuters.fileids()
    except LookupError:
        nltk.download('reuters')

    # Enhanced category mapping with more terms
    category_mapping = {
        # Corporate/Industrial - earnings, acquisitions, jobs
        'earn': 'Corporate_Industrial',
        'acq': 'Corporate_Industrial',
        'ship': 'Corporate_Industrial',
        'jobs': 'Corporate_Industrial',
        'alum': 'Corporate_Industrial',
        'copper': 'Corporate_Industrial',
        'iron-steel': 'Corporate_Industrial',

        # Markets - commodities, forex, precious metals
        'crude': 'Markets',
        'money-fx': 'Markets',
        'wheat': 'Markets',
        'corn': 'Markets',
        'grain': 'Markets',
        'gold': 'Markets',
        'nat-gas': 'Markets',
        'silver': 'Markets',
        'platinum': 'Markets',
        'cocoa': 'Markets',
        'coffee': 'Markets',
        'sugar': 'Markets',

        # Economics - macro indicators, trade, interest rates
        'trade': 'Economics',
        'interest': 'Economics',
        'gnp': 'Economics',
        'cpi': 'Economics',
        'retail': 'Economics',
        'reserves': 'Economics',
        'ipi': 'Economics',
        'money-supply': 'Economics',
        'gdp': 'Economics',
    }

    corpus_data = []
    category_counts = Counter()

    for cat in config.TARGET_CATEGORIES.values():
        category_counts[cat] = 0

    print("\nExtracting documents...")
    print(f"Target: {config.DOCS_PER_CATEGORY} docs per category\n")

    for file_id in reuters.fileids():
        doc_categories = reuters.categories(file_id)

        # Find first matching category that needs more docs
        mapped_category = None
        for cat in doc_categories:
            if cat in category_mapping:
                potential_cat = category_mapping[cat]
                if category_counts[potential_cat] < config.DOCS_PER_CATEGORY:
                    mapped_category = potential_cat
                    break

        if mapped_category:
            text = reuters.raw(file_id)

            corpus_data.append({
                'doc_id': file_id,
                'category': mapped_category,
                'text': text,
                'original_categories': ','.join(doc_categories),
                'word_count': len(text.split())
            })

            category_counts[mapped_category] += 1

            if sum(category_counts.values()) % 200 == 0:
                print(f"Progress: {dict(category_counts)}")

        if all(count >= config.DOCS_PER_CATEGORY for count in category_counts.values()):
            break

    df = pd.DataFrame(corpus_data)

    print(f"\n✓ Created corpus with {len(df)} documents")
    print(f"\nFinal category distribution:")
    print(df['category'].value_counts())

    return df


# ============================================================================
# SOLUTION 2: 4 CATEGORIES (sklearn RCV1)
# ============================================================================

def solution2_sklearn_rcv1(config):
    """
    Use sklearn's RCV1 dataset with proper 4 categories.

    ADVANTAGES:
    - Full 4 categories including Government/Social
    - Larger corpus (800K+ documents available)
    - Hierarchical category structure

    DISADVANTAGES:
    - Sparse matrix format (need reconstruction)
    - Takes longer to process
    """
    from sklearn.datasets import fetch_rcv1

    print("=" * 70)
    print("SOLUTION 2: sklearn RCV1 with 4 Categories")
    print("=" * 70)
    print("Downloading RCV1 (this may take 5-10 minutes)...")

    rcv1 = fetch_rcv1(subset='all', shuffle=False)

    print(f"✓ Downloaded: {rcv1.data.shape[0]:,} documents")

    # Get category indices
    target_names = rcv1.target_names
    category_indices = {}

    for cat_code in config.TARGET_CATEGORIES.keys():
        # Find all subcategories starting with this code
        matching = [i for i, name in enumerate(target_names) if name.startswith(cat_code)]
        category_indices[cat_code] = matching
        print(f"\n{cat_code}: Found {len(matching)} subcategories")

    # Select documents for each category
    selected_docs = {}

    for cat_code, indices in category_indices.items():
        cat_name = config.TARGET_CATEGORIES[cat_code]

        # Get documents belonging to this category
        doc_mask = rcv1.target[:, indices].toarray().sum(axis=1) > 0
        doc_ids = np.where(doc_mask)[0]

        print(f"\n{cat_name}: {len(doc_ids):,} documents available")

        # Sample N documents
        if len(doc_ids) > config.DOCS_PER_CATEGORY:
            np.random.seed(config.RANDOM_SEED)
            selected = np.random.choice(doc_ids, config.DOCS_PER_CATEGORY, replace=False)
        else:
            selected = doc_ids

        selected_docs[cat_name] = selected
        print(f"  → Selected: {len(selected)} documents")

    # Create DataFrame
    # NOTE: sklearn RCV1 doesn't have full text, only bag-of-words
    # For production, you'd need to download original XML files

    corpus_data = []
    for cat_name, doc_indices in selected_docs.items():
        for idx in doc_indices:
            # Get feature representation
            doc_vector = rcv1.data[idx].toarray().flatten()
            top_features = np.argsort(doc_vector)[-50:][::-1]

            # In real scenario, fetch full text from Reuters website
            # For now, use document ID as placeholder
            corpus_data.append({
                'doc_id': f'rcv1_{idx}',
                'category': cat_name,
                'text': f'RCV1_document_{idx}',  # PLACEHOLDER
                'original_index': int(idx),
                'feature_count': len(np.nonzero(doc_vector)[0])
            })

    df = pd.DataFrame(corpus_data)

    print(f"\n✓ Created corpus with {len(df)} documents")
    print(f"\nCategory distribution:")
    print(df['category'].value_counts())

    print("\n⚠ NOTE: sklearn RCV1 doesn't include full text.")
    print("For production use, download original Reuters XML files.")

    return df


# ============================================================================
# SOLUTION 3: ALTERNATIVE - REFRAME AS 3-CATEGORY PROBLEM
# ============================================================================

def solution3_merge_categories(config):
    """
    Merge Government/Social with Economics.

    JUSTIFICATION:
    "We consolidated Economics and Government/Social into a single
    'Economics & Policy' category as these domains exhibit significant
    semantic overlap in news coverage (e.g., fiscal policy, trade agreements)."
    """
    from nltk.corpus import reuters
    import nltk

    print("=" * 70)
    print("SOLUTION 3: 3 Categories with Merged Economics+Government")
    print("=" * 70)

    try:
        reuters.fileids()
    except LookupError:
        nltk.download('reuters')

    # Expanded mapping
    category_mapping = {
        # Corporate/Industrial
        'earn': 'Corporate_Industrial',
        'acq': 'Corporate_Industrial',
        'ship': 'Corporate_Industrial',
        'jobs': 'Corporate_Industrial',
        'alum': 'Corporate_Industrial',

        # Markets
        'crude': 'Markets',
        'money-fx': 'Markets',
        'wheat': 'Markets',
        'corn': 'Markets',
        'grain': 'Markets',
        'gold': 'Markets',

        # Economics & Policy (merged)
        'trade': 'Economics_Policy',
        'interest': 'Economics_Policy',
        'gnp': 'Economics_Policy',
        'cpi': 'Economics_Policy',
        'dlr': 'Economics_Policy',
        'bop': 'Economics_Policy',
        'reserves': 'Economics_Policy',
        'money-supply': 'Economics_Policy',
    }

    # Same logic as Solution 1
    corpus_data = []
    category_counts = Counter({
        'Corporate_Industrial': 0,
        'Markets': 0,
        'Economics_Policy': 0
    })

    target_per_cat = 800

    print(f"\nTarget: {target_per_cat} docs per category\n")

    for file_id in reuters.fileids():
        doc_categories = reuters.categories(file_id)

        mapped_category = None
        for cat in doc_categories:
            if cat in category_mapping:
                potential_cat = category_mapping[cat]
                if category_counts[potential_cat] < target_per_cat:
                    mapped_category = potential_cat
                    break

        if mapped_category:
            text = reuters.raw(file_id)

            corpus_data.append({
                'doc_id': file_id,
                'category': mapped_category,
                'text': text,
                'original_categories': ','.join(doc_categories),
                'word_count': len(text.split())
            })

            category_counts[mapped_category] += 1

            if sum(category_counts.values()) % 200 == 0:
                print(f"Progress: {dict(category_counts)}")

        if all(count >= target_per_cat for count in category_counts.values()):
            break

    df = pd.DataFrame(corpus_data)

    print(f"\n✓ Created corpus with {len(df)} documents")
    print(f"\nCategory distribution:")
    print(df['category'].value_counts())

    return df


# ============================================================================
# SAVE AND STATISTICS
# ============================================================================

def save_corpus_and_stats(df, config, solution_name):
    """Save corpus and generate statistics"""

    Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Create category mapping (alphabetically sorted for consistency)
    unique_categories = sorted(df['category'].unique())
    category_to_id = {cat: idx for idx, cat in enumerate(unique_categories)}
    id_to_category = {idx: cat for cat, idx in category_to_id.items()}

    # Add numeric target column
    df['target'] = df['category'].map(category_to_id)

    # Reorder columns: doc_id, category, target, text, ...
    cols = ['doc_id', 'category', 'target', 'text'] + [c for c in df.columns if c not in ['doc_id', 'category', 'target', 'text']]
    df = df[cols]

    print("\n" + "=" * 70)
    print("CATEGORY MAPPING")
    print("=" * 70)
    for cat, idx in category_to_id.items():
        count = len(df[df['category'] == cat])
        print(f"  {idx} → {cat:30s} ({count} documents)")

    # Save category mapping as JSON
    category_mapping_path = os.path.join(config.OUTPUT_DIR, 'category_mapping.json')
    mapping_data = {
        'category_to_id': category_to_id,
        'id_to_category': id_to_category,
        'description': 'Mapping between category names and numeric targets',
        'num_categories': len(unique_categories),
        'categories_list': unique_categories
    }

    with open(category_mapping_path, 'w') as f:
        json.dump(mapping_data, f, indent=2)

    print(f"\n✓ Category mapping saved to: {category_mapping_path}")

    # Save corpus
    corpus_path = os.path.join(config.OUTPUT_DIR, config.CORPUS_FILE)
    df.to_csv(corpus_path, index=False,sep=';')
    print(f"✓ Corpus saved to: {corpus_path}")

    # Compute statistics
    stats = {
        'solution': solution_name,
        'total_documents': len(df),
        'categories': df['category'].value_counts().to_dict(),
        'category_mapping': category_to_id,
        'num_categories': len(df['category'].unique()),
        'avg_word_count': float(df['word_count'].mean()) if 'word_count' in df.columns else None,
        'target_distribution': df['target'].value_counts().to_dict()
    }

    stats_path = os.path.join(config.OUTPUT_DIR, config.STATS_FILE)
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"✓ Statistics saved to: {stats_path}")

    # Create train/test split
    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df['category'],
        random_state=config.RANDOM_SEED
    )

    train_path = os.path.join(config.OUTPUT_DIR, 'reuters_train.csv')
    test_path = os.path.join(config.OUTPUT_DIR, 'reuters_test.csv')

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"✓ Train: {train_path} ({len(train_df)} docs)")
    print(f"✓ Test: {test_path} ({len(test_df)} docs)")

    return stats


# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Run the selected solution.
    """
    print("\n" + "=" * 70)
    print("REUTERS CORPUS PROCESSING - IMPROVED VERSION")
    print("=" * 70)

    print("\nAvailable solutions:")
    print("1. NLTK Reuters with 3 balanced categories (RECOMMENDED)")
    print("   - 800 docs × 3 categories = 2,400 total")
    print("   - Full text, quick processing")
    print("   - Categories: Corporate/Industrial, Economics, Markets")
    print()
    print("2. sklearn RCV1 with 4 categories (if you have time)")
    print("   - 600 docs × 4 categories = 2,400 total")
    print("   - Includes Government/Social")
    print("   - Requires text reconstruction")
    print()
    print("3. Merged categories (alternative)")
    print("   - 800 docs × 3 categories = 2,400 total")
    print("   - Economics + Government merged as 'Economics & Policy'")

    # AUTO-SELECT SOLUTION 1 for easiest path
    print("\n" + "=" * 70)
    print("AUTO-SELECTING SOLUTION 1 (most practical)")
    print("=" * 70)

    config = ConfigSolution1()
    df = solution1_nltk_3categories(config)
    stats = save_corpus_and_stats(df, config, "Solution 1: 3 Categories")

    print("\n" + "=" * 70)
    print("✓ PROCESSING COMPLETE!")
    print("=" * 70)

    print("\n📊 Summary:")
    print(f"  - Total documents: {len(df)}")
    print(f"  - Categories: {len(df['category'].unique())}")
    print(f"  - Average words/doc: {df['word_count'].mean():.0f}")

    print("\n📁 Files created:")
    print(f"  1. {config.OUTPUT_DIR}/reuters_dataset_raw.csv")
    print(f"  2. {config.OUTPUT_DIR}/reuters_train.csv")
    print(f"  3. {config.OUTPUT_DIR}/reuters_test.csv")
    print(f"  4. {config.OUTPUT_DIR}/reuters_statistics.json")
    print(f"  5. {config.OUTPUT_DIR}/category_mapping.json ⭐ NEW")

    print("\n🎯 Category Mapping:")
    print("  " + json.dumps(stats['category_mapping'], indent=4).replace('\n', '\n  '))

    print("\n💡 For reviewer response, use this justification:")
    print("""
    "We selected the three most well-represented categories in Reuters
    (Corporate/Industrial, Economics, Markets) to ensure balanced evaluation.
    While Government/Social topics exist in Reuters, they are sparsely 
    represented in the NLTK version. Our 2,400-document corpus (800 per
    category) provides robust cross-domain validation alongside Amazon 
    and BBC corpora."
    """)

    print("\n" + "=" * 70)
    print("Next steps:")
    print("1. Verify: reuters_processed/reuters_dataset_raw.csv")
    print("2. Run OBOE pipeline on this corpus")
    print("3. Update tables with '3 categories' instead of '4'")
    print("4. Use justification above in reviewer response")
    print("=" * 70)

    return df, stats


if __name__ == "__main__":
    df, stats = main()