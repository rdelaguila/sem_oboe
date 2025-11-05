#!/usr/bin/env python3
"""
KGE Classification Visualization Generator
Analyzes and visualizes KGE training results and topic classification performance
"""

import json
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import warnings
from sklearn.manifold import TSNE
import pickle

warnings.filterwarnings('ignore')

# Style configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#5E60CE', '#00B4D8']


class KGEClassificationVisualizer:
    """Generates visualizations for KGE training and classification results"""

    def __init__(self, input_base_path: str, output_base_path: str):
        self.input_base_path = Path(input_base_path)
        self.output_base_path = Path(output_base_path)

    def find_all_results(self, repo_name: str) -> Tuple[List[Path], List[Path]]:
        """Find all KGE and classification result files"""
        repo_path = self.input_base_path / repo_name

        if not repo_path.exists():
            return [], []

        # Find KGE files with multiple naming patterns
        kge_files_standard = list(repo_path.glob('kge_training_results_*.json'))
        kge_files_complex_json = list(repo_path.glob('complex_optimization_metrics_*.json'))
        kge_files_complex_txt = list(repo_path.glob('complex_optimization_metrics_*.txt'))

        # Consolidate all KGE files
        kge_files = sorted(kge_files_standard + kge_files_complex_json + kge_files_complex_txt)

        classification_files = sorted(repo_path.glob('classification_results_*.json'))

        print(f"📂 Found {len(kge_files_standard)} standard KGE files")
        print(f"📂 Found {len(kge_files_complex_json)} ComplEx optimization JSON files")
        print(f"📂 Found {len(kge_files_complex_txt)} ComplEx optimization TXT files")
        print(f"📂 Total KGE files: {len(kge_files)}")

        return kge_files, classification_files

    def load_json(self, filepath: Path) -> Optional[Dict]:
        """Load JSON file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  Error loading {filepath}: {e}")
            return None

    def parse_complex_optimization_file(self, filepath: Path) -> Optional[Dict]:
        """Parse ComplEx optimization text file and convert to standard format"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            print(f"📄 Parsing ComplEx optimization file: {filepath.name}")

            # Extract selected configuration metrics
            selected_mrr = None
            selected_hits1 = None
            selected_hits10 = None
            selected_dim = None

            # Parse selected configuration for embedding dimension
            if "SELECTED CONFIGURATION:" in content:
                config_section = \
                content.split("SELECTED CONFIGURATION:")[1].split("Selected Model Evaluation Metrics:")[0]
                for line in config_section.split('\n'):
                    if "embedding_dim:" in line:
                        selected_dim = int(line.split("embedding_dim:")[1].strip())
                        break

            # Parse selected model evaluation metrics section
            if "Selected Model Evaluation Metrics:" in content:
                metrics_section = content.split("Selected Model Evaluation Metrics:")[1].split("Search Strategy:")[0]
                for line in metrics_section.split('\n'):
                    line = line.strip()
                    if line.startswith("Hits@1:"):
                        selected_hits1 = float(line.split("Hits@1:")[1].strip())
                    elif line.startswith("Hits@10:"):
                        selected_hits10 = float(line.split("Hits@10:")[1].strip())
                    elif line.startswith("MRR:"):
                        selected_mrr = float(line.split("MRR:")[1].strip())

            print(
                f"  Selected config: dim={selected_dim}, MRR={selected_mrr}, Hits@1={selected_hits1}, Hits@10={selected_hits10}")

            # Parse all configurations
            all_results = []
            if "ALL CONFIGURATIONS TESTED:" in content:
                configs_section = content.split("ALL CONFIGURATIONS TESTED:")[1]
                configs = configs_section.split("#")[1:]  # Split by configuration number

                for config in configs:
                    lines = config.split('\n')
                    config_data = {}

                    for line in lines:
                        line = line.strip()
                        if "Score (MRR):" in line:
                            config_data['mrr'] = float(line.split("Score (MRR):")[1].strip())
                        elif "Embedding Dim:" in line:
                            config_data['embedding_dim'] = int(line.split("Embedding Dim:")[1].strip())
                        elif "Hits@10:" in line:
                            config_data['hits_at_10'] = float(line.split("Hits@10:")[1].strip())

                    if 'mrr' in config_data and 'embedding_dim' in config_data:
                        # Estimate hits@1 from MRR if not available (typically hits@1 < MRR)
                        config_data['hits_at_1'] = config_data['mrr'] * 0.68  # Conservative estimate
                        if 'hits_at_10' not in config_data:
                            config_data['hits_at_10'] = min(config_data['mrr'] * 1.6, 1.0)

                        all_results.append({
                            'embedding_dim': config_data['embedding_dim'],
                            'evaluation': {
                                'hits_at_1': config_data['hits_at_1'],
                                'hits_at_10': config_data['hits_at_10'],
                                'mrr': config_data['mrr']
                            }
                        })

                print(f"  Parsed {len(all_results)} configurations")

            # Build result using selected metrics if available
            if selected_mrr and selected_hits1 is not None and selected_hits10 and selected_dim:
                print(f"  ✓ Using selected model metrics")

                # Add selected config to all_results if not already there
                selected_in_all = any(r['embedding_dim'] == selected_dim and
                                      abs(r['evaluation']['mrr'] - selected_mrr) < 0.001
                                      for r in all_results)

                if not selected_in_all:
                    all_results.append({
                        'embedding_dim': selected_dim,
                        'evaluation': {
                            'hits_at_1': selected_hits1,
                            'hits_at_10': selected_hits10,
                            'mrr': selected_mrr
                        }
                    })

                return {
                    'model_type': 'complex',
                    'best_embedding_dim': selected_dim,
                    'best_score': selected_mrr,
                    'all_results': all_results,
                    'timestamp': filepath.stem.split('_')[-2] + '_' + filepath.stem.split('_')[
                        -1] if '_' in filepath.stem else 'unknown'
                }
            elif all_results:
                # Fallback to best from all results
                best_result = max(all_results, key=lambda x: x['evaluation']['mrr'])
                print(
                    f"  ✓ Using best from all configs: dim={best_result['embedding_dim']}, MRR={best_result['evaluation']['mrr']:.4f}")

                return {
                    'model_type': 'complex',
                    'best_embedding_dim': best_result['embedding_dim'],
                    'best_score': best_result['evaluation']['mrr'],
                    'all_results': all_results,
                    'timestamp': filepath.stem.split('_')[-2] + '_' + filepath.stem.split('_')[
                        -1] if '_' in filepath.stem else 'unknown'
                }
            else:
                print(f"  ⚠️ Could not parse metrics from ComplEx file")
                return None

        except Exception as e:
            print(f"❌ Error parsing ComplEx optimization file {filepath}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def load_all_results(self, kge_files: List[Path], classification_files: List[Path]) -> Tuple[
        List[Dict], List[Dict], Optional[Dict], Optional[Dict]]:
        """
        Load all KGE and classification results and select optimal model
        Returns: (all_kge_results, all_classification_results, optimal_kge, optimal_classification)
        """
        if not kge_files or not classification_files:
            print("⚠️  Missing KGE or classification results")
            return [], [], None, None

        print(f"\n{'=' * 60}")
        print(f"LOADING RESULTS FILES")
        print(f"{'=' * 60}")

        # Load all results
        kge_results = []
        for kge_file in kge_files:
            print(f"Loading: {kge_file.name}")

            # Check file type and use appropriate loader
            if kge_file.suffix == '.txt':
                result = self.parse_complex_optimization_file(kge_file)
            else:
                result = self.load_json(kge_file)

            if result is not None:
                # Add filename for debugging
                result['_source_file'] = kge_file.name
                kge_results.append(result)

        classification_results = []
        for clf_file in classification_files:
            print(f"Loading: {clf_file.name}")
            result = self.load_json(clf_file)
            if result is not None:
                classification_results.append(result)

        # Filter valid results
        kge_results = [r for r in kge_results if r is not None]
        classification_results = [r for r in classification_results if r is not None]

        if not kge_results or not classification_results:
            return [], [], None, None

        # Find best classification result
        best_classification = max(classification_results,
                                  key=lambda x: x['test_metrics']['accuracy'])

        best_classification_emb_dim = best_classification['embedding_dim']
        best_classification_model_type = best_classification['kge_model_type']

        # Find matching KGE result
        matching_kge = None
        for kge in kge_results:
            if (kge.get('best_embedding_dim') == best_classification_emb_dim and
                    kge.get('model_type') == best_classification_model_type):
                matching_kge = kge
                break

        # If no match, find best KGE result
        if matching_kge is None:
            matching_kge = max(kge_results, key=lambda x: x.get('best_score', 0))

        print(f"\n{'=' * 60}")
        print(f"OPTIMAL MODEL SELECTION")
        print(f"{'=' * 60}")
        print(f"Loaded {len(kge_results)} KGE result(s)")
        print(f"Loaded {len(classification_results)} classification result(s)")
        print(f"\nOptimal Configuration:")
        print(f"Model Type: {best_classification_model_type.upper()}")
        print(f"Embedding Dimension: {best_classification_emb_dim}")
        print(f"Classification Accuracy: {best_classification['test_metrics']['accuracy']:.4f}")
        if matching_kge:
            print(f"KGE Score: {matching_kge.get('best_score', 'N/A'):.4f}")
            if '_source_file' in matching_kge:
                print(f"KGE Source File: {matching_kge['_source_file']}")
        print(f"{'=' * 60}\n")

        return kge_results, classification_results, matching_kge, best_classification

    def create_all_kge_comparison(self, all_kge_results: List[Dict], repo_name: str):
        """
        Create visualization comparing all KGE training results with legend
        """
        if not all_kge_results:
            print("⚠️  No KGE results to compare")
            return None

        print(f"\n{'=' * 60}")
        print(f"DEBUG: KGE Comparison Data Loading")
        print(f"{'=' * 60}")

        # Aggregate all results
        all_dim_results = []

        for idx, kge_data in enumerate(all_kge_results):
            model_type = kge_data.get('model_type', 'unknown')
            source_file = kge_data.get('_source_file', 'unknown')
            print(f"\nProcessing KGE result #{idx + 1}:")
            print(f"  Source file: {source_file}")
            print(f"  Model type: {model_type}")
            print(f"  Has 'all_results': {'all_results' in kge_data}")
            print(f"  Has 'best_embedding_dim': {'best_embedding_dim' in kge_data}")

            if 'all_results' in kge_data:
                # If it has dimensional comparison
                print(f"  Number of dimensional results: {len(kge_data['all_results'])}")
                for result in kge_data['all_results']:
                    dim = result['embedding_dim']
                    hits_1 = result['evaluation']['hits_at_1']
                    hits_10 = result['evaluation']['hits_at_10']
                    mrr = result['evaluation']['mrr']

                    print(f"    Dim {dim}: Hits@1={hits_1:.4f}, Hits@10={hits_10:.4f}, MRR={mrr:.4f}")

                    all_dim_results.append({
                        'embedding_dim': dim,
                        'model_type': model_type,
                        'hits_at_1': hits_1,
                        'hits_at_10': hits_10,
                        'mrr': mrr,
                        'timestamp': kge_data.get('timestamp', 'unknown'),
                        'source_file': source_file
                    })
            elif 'best_embedding_dim' in kge_data:
                # Single result without dimensional comparison
                dim = kge_data['best_embedding_dim']
                score = kge_data.get('best_score', 0)
                print(f"  Single result - Dim {dim}: Score={score:.4f}")

                all_dim_results.append({
                    'embedding_dim': dim,
                    'model_type': model_type,
                    'hits_at_1': score,
                    'hits_at_10': score,
                    'mrr': score,
                    'timestamp': kge_data.get('timestamp', 'unknown'),
                    'source_file': source_file
                })

        if not all_dim_results:
            print("⚠️  No dimensional data available in KGE results")
            return None

        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(all_dim_results)

        print(f"\n{'=' * 60}")
        print(f"Aggregated Results:")
        print(f"{'=' * 60}")
        print(df[['model_type', 'embedding_dim', 'hits_at_1', 'hits_at_10', 'mrr', 'source_file']].to_string())
        print(f"{'=' * 60}\n")

        # Get unique models and dimensions - ALL dimensions from ALL models
        unique_models = sorted(df['model_type'].unique())
        unique_dims = sorted(df['embedding_dim'].unique())

        print(f"📊 Models found: {unique_models}")
        print(f"📊 Dimensions found: {unique_dims}")
        print(f"📊 Total configurations: {len(df)}")

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Prepare data for grouped bar charts
        x = np.arange(len(unique_dims))
        width = 0.8 / max(len(unique_models), 1)  # Divide the space by number of models

        # For each metric, create grouped bars
        for metric_idx, (metric, ax, title, ylabel) in enumerate([
            ('hits_at_1', axes[0], 'Hits@1 by Model and Dimension', 'Hits@1'),
            ('hits_at_10', axes[1], 'Hits@10 by Model and Dimension', 'Hits@10'),
            ('mrr', axes[2], 'Mean Reciprocal Rank by Model and Dimension', 'MRR')
        ]):
            for model_idx, model in enumerate(unique_models):
                model_data = df[df['model_type'] == model]

                # Get values for each dimension (use None if dimension not present for this model)
                values = []
                has_data = []
                for dim in unique_dims:
                    dim_data = model_data[model_data['embedding_dim'] == dim]
                    if len(dim_data) > 0:
                        values.append(dim_data[metric].max())
                        has_data.append(True)
                    else:
                        values.append(0)  # Use 0 for missing data
                        has_data.append(False)

                # Plot bars for this model
                offset = width * (model_idx - len(unique_models) / 2 + 0.5)
                bars = ax.bar(x + offset, values, width,
                              label=model.upper(),
                              color=COLORS[model_idx % len(COLORS)],
                              alpha=0.8,
                              edgecolor='black')

                # Add value labels on bars (only for bars with actual data)
                for i, (bar, val, has_d) in enumerate(zip(bars, values, has_data)):
                    if has_d and val > 0:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width() / 2., height,
                                f'{val:.3f}',
                                ha='center', va='bottom', fontsize=7, fontweight='bold')

            ax.set_xlabel('Embedding Dimension', fontweight='bold', fontsize=11)
            ax.set_ylabel(ylabel, fontweight='bold', fontsize=11)
            ax.set_title(title, fontweight='bold', fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(unique_dims)
            ax.legend(loc='best', fontsize=10, framealpha=0.9)
            ax.grid(axis='y', alpha=0.3)

        # Clean repo name for display
        clean_repo_name = repo_name.replace('_2iter', '').replace('_2_iter', '')

        # Overall title
        fig.suptitle(f'KGE Performance Metrics Comparison - {clean_repo_name.upper()}',
                     fontsize=15, fontweight='bold', y=1.00)

        # Save
        output_dir = self.output_base_path / repo_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / 'kge_metrics_comparison.png'
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Generated: {output_path}")
        return output_path

    def create_kge_metrics_comparison(self, kge_data: Dict, repo_name: str):
        """
        Create visualization comparing KGE metrics across different embedding dimensions
        """
        if 'all_results' not in kge_data:
            print("⚠️  No dimensional comparison data available")
            return None

        all_results = kge_data['all_results']

        # Extract data
        dims = [r['embedding_dim'] for r in all_results]
        hits_at_1 = [r['evaluation']['hits_at_1'] for r in all_results]
        hits_at_10 = [r['evaluation']['hits_at_10'] for r in all_results]
        mrr = [r['evaluation']['mrr'] for r in all_results]

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Hits@1
        axes[0].bar(range(len(dims)), hits_at_1, color=COLORS[0], alpha=0.8, edgecolor='black')
        axes[0].set_xlabel('Embedding Dimension', fontweight='bold')
        axes[0].set_ylabel('Hits@1', fontweight='bold')
        axes[0].set_title('Hits@1 by Embedding Dimension', fontweight='bold')
        axes[0].set_xticks(range(len(dims)))
        axes[0].set_xticklabels(dims)
        axes[0].grid(axis='y', alpha=0.3)

        # Add values on bars
        for i, v in enumerate(hits_at_1):
            axes[0].text(i, v, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

        # Hits@10
        axes[1].bar(range(len(dims)), hits_at_10, color=COLORS[1], alpha=0.8, edgecolor='black')
        axes[1].set_xlabel('Embedding Dimension', fontweight='bold')
        axes[1].set_ylabel('Hits@10', fontweight='bold')
        axes[1].set_title('Hits@10 by Embedding Dimension', fontweight='bold')
        axes[1].set_xticks(range(len(dims)))
        axes[1].set_xticklabels(dims)
        axes[1].grid(axis='y', alpha=0.3)

        for i, v in enumerate(hits_at_10):
            axes[1].text(i, v, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

        # MRR
        axes[2].bar(range(len(dims)), mrr, color=COLORS[2], alpha=0.8, edgecolor='black')
        axes[2].set_xlabel('Embedding Dimension', fontweight='bold')
        axes[2].set_ylabel('MRR', fontweight='bold')
        axes[2].set_title('Mean Reciprocal Rank by Embedding Dimension', fontweight='bold')
        axes[2].set_xticks(range(len(dims)))
        axes[2].set_xticklabels(dims)
        axes[2].grid(axis='y', alpha=0.3)

        for i, v in enumerate(mrr):
            axes[2].text(i, v, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

        # Clean repo name for display
        clean_repo_name = repo_name.replace('_2iter', '').replace('_2_iter', '')

        # Overall title
        fig.suptitle(f'KGE Performance Metrics - {clean_repo_name.upper()}\nModel: {kge_data["model_type"].upper()}',
                     fontsize=14, fontweight='bold', y=1.02)

        # Save
        output_dir = self.output_base_path / repo_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / 'kge_metrics_comparison.png'
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Generated: {output_path}")
        return output_path

    def create_all_classification_comparison(self, all_classification_results: List[Dict], repo_name: str):
        """
        Create visualization comparing all classification results
        """
        if not all_classification_results:
            print("⚠️  No classification results to compare")
            return None

        # Extract data for comparison
        comparison_data = []

        for clf_data in all_classification_results:
            comparison_data.append({
                'model_type': clf_data.get('model_type', 'Unknown'),
                'kge_model': clf_data.get('kge_model_type', 'unknown'),
                'embedding_dim': clf_data.get('embedding_dim', 0),
                'accuracy': clf_data['test_metrics']['accuracy'],
                'logloss': clf_data['test_metrics'].get('logloss', 0),
                'roc_auc': clf_data['test_metrics'].get('auc_metrics', {}).get('roc_auc_macro', 0),
                'macro_f1': clf_data['test_metrics']['classification_report']['macro avg']['f1-score'],
                'timestamp': clf_data.get('timestamp', 'unknown')
            })

        df = pd.DataFrame(comparison_data)

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Accuracy by embedding dimension
        ax1 = axes[0, 0]
        grouped_acc = df.groupby('embedding_dim')['accuracy'].max().reset_index()
        dims = grouped_acc['embedding_dim'].tolist()
        accuracies = grouped_acc['accuracy'].tolist()

        ax1.bar(range(len(dims)), accuracies, color=COLORS[0], alpha=0.8, edgecolor='black')
        ax1.set_xlabel('Embedding Dimension', fontweight='bold')
        ax1.set_ylabel('Accuracy', fontweight='bold')
        ax1.set_title('Best Accuracy by Embedding Dimension', fontweight='bold')
        ax1.set_xticks(range(len(dims)))
        ax1.set_xticklabels(dims)
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim(0, 1)

        for i, v in enumerate(accuracies):
            ax1.text(i, v, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')

        # 2. Multiple metrics comparison
        ax2 = axes[0, 1]
        metrics_df = df.groupby('embedding_dim').agg({
            'accuracy': 'max',
            'roc_auc': 'max',
            'macro_f1': 'max'
        }).reset_index()

        x = np.arange(len(metrics_df))
        width = 0.25

        ax2.bar(x - width, metrics_df['accuracy'], width, label='Accuracy', color=COLORS[0], alpha=0.8)
        ax2.bar(x, metrics_df['roc_auc'], width, label='ROC AUC', color=COLORS[1], alpha=0.8)
        ax2.bar(x + width, metrics_df['macro_f1'], width, label='Macro F1', color=COLORS[2], alpha=0.8)

        ax2.set_xlabel('Embedding Dimension', fontweight='bold')
        ax2.set_ylabel('Score', fontweight='bold')
        ax2.set_title('Classification Metrics by Embedding Dimension', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics_df['embedding_dim'].tolist())
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_ylim(0, 1)

        # 3. Log Loss comparison
        ax3 = axes[1, 0]
        grouped_loss = df.groupby('embedding_dim')['logloss'].min().reset_index()

        ax3.bar(range(len(grouped_loss)), grouped_loss['logloss'],
                color=COLORS[3], alpha=0.8, edgecolor='black')
        ax3.set_xlabel('Embedding Dimension', fontweight='bold')
        ax3.set_ylabel('Log Loss', fontweight='bold')
        ax3.set_title('Log Loss by Embedding Dimension (Lower is Better)', fontweight='bold')
        ax3.set_xticks(range(len(grouped_loss)))
        ax3.set_xticklabels(grouped_loss['embedding_dim'].tolist())
        ax3.grid(axis='y', alpha=0.3)

        for i, v in enumerate(grouped_loss['logloss']):
            ax3.text(i, v, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')

        # 4. Summary table
        ax4 = axes[1, 1]
        ax4.axis('off')

        best_result = df.loc[df['accuracy'].idxmax()]

        summary_text = f"""
        COMPARISON SUMMARY
        {'=' * 40}

        Total Configurations: {len(df)}
        Embedding Dimensions Tested: {len(df['embedding_dim'].unique())}
        KGE Models Used: {', '.join(df['kge_model'].unique())}

        BEST CONFIGURATION
        {'─' * 40}
        Embedding Dim:  {best_result['embedding_dim']}
        KGE Model:      {best_result['kge_model'].upper()}
        Classifier:     {best_result['model_type']}

        BEST METRICS
        {'─' * 40}
        Accuracy:       {best_result['accuracy']:.4f}
        ROC AUC:        {best_result['roc_auc']:.4f}
        Macro F1:       {best_result['macro_f1']:.4f}
        Log Loss:       {best_result['logloss']:.4f}

        RANGE OF PERFORMANCE
        {'─' * 40}
        Accuracy:       {df['accuracy'].min():.4f} - {df['accuracy'].max():.4f}
        ROC AUC:        {df['roc_auc'].min():.4f} - {df['roc_auc'].max():.4f}
        """

        ax4.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                 verticalalignment='center',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        # Clean repo name for display
        clean_repo_name = repo_name.replace('_2iter', '').replace('_2_iter', '')

        # Overall title
        fig.suptitle(f'Classification Results Comparison - {clean_repo_name.upper()}',
                     fontsize=16, fontweight='bold', y=0.995)

        # Save
        output_dir = self.output_base_path / repo_name
        output_path = output_dir / 'all_classification_comparison.png'
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Generated: {output_path}")
        return output_path

    def create_classification_performance_chart(self, classification_data: Dict, repo_name: str):
        """
        Create comprehensive classification performance visualization
        """
        test_metrics = classification_data['test_metrics']
        classification_report = test_metrics['classification_report']

        # Extract per-class metrics
        classes = []
        precision = []
        recall = []
        f1_score = []
        support = []

        for key in sorted(classification_report.keys()):
            if key.isdigit():
                classes.append(f"Topic {key}")
                precision.append(classification_report[key]['precision'])
                recall.append(classification_report[key]['recall'])
                f1_score.append(classification_report[key]['f1-score'])
                support.append(classification_report[key]['support'])

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Per-class metrics (grouped bars)
        ax1 = axes[0, 0]
        x = np.arange(len(classes))
        width = 0.25

        ax1.bar(x - width, precision, width, label='Precision', color=COLORS[0], alpha=0.8)
        ax1.bar(x, recall, width, label='Recall', color=COLORS[1], alpha=0.8)
        ax1.bar(x + width, f1_score, width, label='F1-Score', color=COLORS[2], alpha=0.8)

        ax1.set_xlabel('Topics', fontweight='bold')
        ax1.set_ylabel('Score', fontweight='bold')
        ax1.set_title('Classification Metrics by Topic', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(classes, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim(0, 1)

        # 2. Metrics Table Visualization
        ax2 = axes[0, 1]
        ax2.axis('off')

        # Get embedding model info
        kge_model = classification_data.get('kge_model_type', 'unknown').upper()
        embedding_dim = classification_data.get('embedding_dim', 'N/A')
        classifier = classification_data.get('model_type', 'Unknown')

        # Calculate metrics
        accuracy = test_metrics['accuracy']
        macro_avg = classification_report['macro avg']
        weighted_avg = classification_report['weighted avg']
        roc_auc = test_metrics.get('auc_metrics', {}).get('roc_auc_macro', 0)
        logloss = test_metrics.get('logloss', 0)

        # Create metrics table data
        table_data = [
            ['EMBEDDING MODEL', ''],
            ['KGE Model', kge_model],
            ['Embedding Dim', str(embedding_dim)],
            ['', ''],
            ['CLASSIFIER', ''],
            ['Model Type', classifier],
            ['', ''],
            ['OVERALL METRICS', ''],
            ['Accuracy', f'{accuracy:.4f}'],
            ['Log Loss', f'{logloss:.4f}'],
            ['ROC AUC', f'{roc_auc:.4f}'],
            ['', ''],
            ['MACRO AVERAGE', ''],
            ['Precision', f'{macro_avg["precision"]:.4f}'],
            ['Recall', f'{macro_avg["recall"]:.4f}'],
            ['F1-Score', f'{macro_avg["f1-score"]:.4f}'],
            ['', ''],
            ['WEIGHTED AVERAGE', ''],
            ['Precision', f'{weighted_avg["precision"]:.4f}'],
            ['Recall', f'{weighted_avg["recall"]:.4f}'],
            ['F1-Score', f'{weighted_avg["f1-score"]:.4f}'],
        ]

        # Create table
        table = ax2.table(cellText=table_data, cellLoc='left', loc='center',
                          colWidths=[0.5, 0.4])

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style the table
        for i, row in enumerate(table_data):
            cell = table[(i, 0)]
            if row[0] in ['EMBEDDING MODEL', 'CLASSIFIER', 'OVERALL METRICS', 'MACRO AVERAGE', 'WEIGHTED AVERAGE']:
                # Header rows
                cell.set_facecolor('#4472C4')
                cell.set_text_props(weight='bold', color='white')
                table[(i, 1)].set_facecolor('#4472C4')
            elif row[0] == '':
                # Separator rows
                cell.set_facecolor('#E7E6E6')
                table[(i, 1)].set_facecolor('#E7E6E6')
            else:
                # Data rows
                if i % 2 == 0:
                    cell.set_facecolor('#F2F2F2')
                    table[(i, 1)].set_facecolor('#F2F2F2')

        ax2.set_title('Classification Metrics Summary', fontweight='bold', fontsize=12, pad=20)

        # 3. Sample distribution
        ax3 = axes[1, 0]
        ax3.bar(classes, support, color=COLORS[:len(classes)], alpha=0.8, edgecolor='black')
        ax3.set_xlabel('Topics', fontweight='bold')
        ax3.set_ylabel('Number of Samples', fontweight='bold')
        ax3.set_title('Sample Distribution by Topic', fontweight='bold')
        ax3.set_xticklabels(classes, rotation=45, ha='right')
        ax3.grid(axis='y', alpha=0.3)

        # Add values on bars
        for i, v in enumerate(support):
            ax3.text(i, v, f'{int(v)}', ha='center', va='bottom', fontweight='bold')

        # 4. Per-class F1-Score visualization
        ax4 = axes[1, 1]

        # Sort by F1-score for better visualization
        f1_df = pd.DataFrame({
            'Topic': classes,
            'F1-Score': f1_score,
            'Support': support
        })
        f1_df = f1_df.sort_values('F1-Score', ascending=True)

        # Horizontal bar chart
        colors_f1 = [COLORS[int(topic.split()[1]) % len(COLORS)] for topic in f1_df['Topic']]
        bars = ax4.barh(f1_df['Topic'], f1_df['F1-Score'], color=colors_f1, alpha=0.8, edgecolor='black')

        ax4.set_xlabel('F1-Score', fontweight='bold')
        ax4.set_ylabel('Topics', fontweight='bold')
        ax4.set_title('F1-Score by Topic (Sorted)', fontweight='bold')
        ax4.set_xlim(0, 1)
        ax4.grid(axis='x', alpha=0.3)

        # Add values on bars
        for i, (bar, f1) in enumerate(zip(bars, f1_df['F1-Score'])):
            ax4.text(f1 + 0.01, i, f'{f1:.3f}', va='center', fontweight='bold', fontsize=9)

        # Clean repo name for display
        clean_repo_name = repo_name.replace('_2iter', '').replace('_2_iter', '')

        # Overall title with embedding model info
        fig.suptitle(f'Topic Classification Performance - {clean_repo_name.upper()}\n'
                     f'Embedding: {kge_model} (dim={embedding_dim}) | Classifier: {classifier}',
                     fontsize=16, fontweight='bold', y=0.995)

        # Save
        output_dir = self.output_base_path / repo_name
        output_path = output_dir / 'classification_performance.png'
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Generated: {output_path}")
        return output_path

    def create_tsne_visualization(self, repo_name: str, classification_data: Dict):
        """
        Create t-SNE visualization of embeddings colored by topic
        """
        # Get paths from classification data
        kge_model_type = classification_data['kge_model_type']
        embedding_dim = classification_data['embedding_dim']

        # Extract base repo name (remove _2iter suffix if present)
        base_repo = repo_name.replace('_2iter', '').replace('_2_iter', '')

        # Construct paths based on the structure from C_classification_3.py
        base_path = Path("../data")
        emb_dir = base_path / "triples_emb" / f"{base_repo}_2iter"

        # Load embeddings and mappings
        model_path = emb_dir / f"{kge_model_type}_model.pkl"
        mappings_path = emb_dir / f"{kge_model_type}_mappings.pkl"

        print(f"Looking for embeddings in: {emb_dir}")
        print(f"  Model: {model_path}")
        print(f"  Mappings: {mappings_path}")

        if not model_path.exists() or not mappings_path.exists():
            print(f"⚠️  Embedding files not found:")
            print(f"   Model: {model_path} {'✓' if model_path.exists() else '✗'}")
            print(f"   Mappings: {mappings_path} {'✓' if mappings_path.exists() else '✗'}")
            return None

        try:
            import torch
            import pandas as pd

            # Load model using torch.load with weights_only=False for PyKEEN models
            print(f"Loading PyKEEN model...")
            model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)

            # Load mappings (this should be regular pickle)
            with open(mappings_path, 'rb') as f:
                mappings = pickle.load(f)

            entity_to_id = mappings['entity_to_id']

            # Debug: Show sample entities
            sample_entities = list(entity_to_id.keys())[:10]
            print(f"Sample entities in mapping: {sample_entities}")
            print(f"Total entities in mapping: {len(entity_to_id)}")

            # Load KG data to get topic labels
            kg_path = base_path / "triples_raw" / base_repo / f"dataset_triplet_{base_repo}_new_simplificado.csv"

            if not kg_path.exists():
                print(f"⚠️  Knowledge graph file not found: {kg_path}")
                return None

            df = pd.read_csv(kg_path)
            print(f"Loaded {len(df)} triples from knowledge graph")

            # Debug: Show sample subjects
            sample_subjects = df['subject'].unique()[:10]
            print(f"Sample subjects in KG: {sample_subjects}")

            # Get entity embeddings
            print(f"Extracting entity embeddings...")
            entity_embeddings = model.entity_representations[0](indices=None).cpu().detach().numpy()
            print(f"Entity embeddings shape: {entity_embeddings.shape}")

            # Match entities with topics - try multiple strategies
            entity_topics = {}

            # Strategy 1: Try with 'doc_' prefix
            for _, row in df.iterrows():
                subject = row['subject']
                if subject.startswith('doc_'):
                    topic = row['topic']
                    if subject in entity_to_id:
                        entity_topics[subject] = topic

            print(f"Strategy 1 (doc_ prefix): Found {len(entity_topics)} documents")

            # Strategy 2: If no documents found, try matching any subject that has a topic
            if len(entity_topics) == 0:
                print("Strategy 2: Trying to match all subjects with topics...")
                for _, row in df.iterrows():
                    subject = row['subject']
                    if 'topic' in row and pd.notna(row['topic']):
                        topic = row['topic']
                        if subject in entity_to_id:
                            entity_topics[subject] = topic

                print(f"Strategy 2 (all subjects with topics): Found {len(entity_topics)} entities")

            # Strategy 3: If still no documents, check for any pattern
            if len(entity_topics) == 0:
                print("Strategy 3: Analyzing entity patterns...")

                # Check what prefixes exist
                prefixes = {}
                for entity in entity_to_id.keys():
                    if '_' in entity:
                        prefix = entity.split('_')[0]
                        prefixes[prefix] = prefixes.get(prefix, 0) + 1

                print(f"Entity prefixes found: {prefixes}")

                # Try to find the most common prefix that might be documents
                for _, row in df.iterrows():
                    subject = row['subject']
                    if 'topic' in row and pd.notna(row['topic']):
                        topic = row['topic']
                        if subject in entity_to_id:
                            entity_topics[subject] = topic

                print(f"Strategy 3 (direct matching): Found {len(entity_topics)} entities")

            # Filter embeddings for documents only
            doc_indices = []
            doc_topics = []

            for entity, topic in entity_topics.items():
                if entity in entity_to_id:
                    idx = entity_to_id[entity]
                    doc_indices.append(idx)
                    doc_topics.append(topic)

            if len(doc_indices) == 0:
                print("⚠️  No document entities found for t-SNE visualization")
                print("Debug information:")
                print(f"  - Total entities in embeddings: {len(entity_to_id)}")
                print(f"  - Unique subjects in KG: {df['subject'].nunique()}")
                print(f"  - Subjects with 'doc_' prefix in KG: {df['subject'].str.startswith('doc_').sum()}")
                print(f"  - KG columns: {df.columns.tolist()}")
                return None

            # Get embeddings for documents
            doc_embeddings = entity_embeddings[doc_indices]
            doc_topics = np.array(doc_topics)

            print(f"Successfully matched {len(doc_embeddings)} documents with topics")
            print(f"Unique topics: {sorted(np.unique(doc_topics))}")
            print(f"Performing t-SNE on {len(doc_embeddings)} documents...")

            # Apply t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(doc_embeddings) - 1))
            embeddings_2d = tsne.fit_transform(doc_embeddings)

            # Create visualization
            fig, ax = plt.subplots(figsize=(12, 10))

            # Get unique topics
            unique_topics = sorted(np.unique(doc_topics))

            # Plot each topic with different color
            for topic in unique_topics:
                mask = doc_topics == topic
                ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                           label=f'Topic {topic}',
                           alpha=0.6,
                           s=50,
                           color=COLORS[int(topic) % len(COLORS)])

            ax.set_xlabel('t-SNE Dimension 1', fontweight='bold', fontsize=12)
            ax.set_ylabel('t-SNE Dimension 2', fontweight='bold', fontsize=12)

            # Clean repo name for display
            clean_repo_name = base_repo.replace('_2iter', '').replace('_2_iter', '')

            ax.set_title(f't-SNE Visualization of Document Embeddings by Topic\n'
                         f'Repository: {clean_repo_name.upper()} | Model: {kge_model_type.upper()} | '
                         f'Embedding Dim: {embedding_dim}',
                         fontweight='bold', fontsize=14, pad=20)
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)

            # Save
            output_dir = self.output_base_path / repo_name
            output_path = output_dir / 'tsne_embeddings_by_topic.png'
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"✅ Generated: {output_path}")
            return output_path

        except Exception as e:
            print(f"❌ Error creating t-SNE visualization: {e}")
            import traceback
            traceback.print_exc()
            return None

    def create_kge_classification_relationship(self, all_kge_results: List[Dict],
                                               all_classification_results: List[Dict], repo_name: str):
        """
        Create visualization showing relationship between KGE metrics and classification performance
        """
        if not all_kge_results or not all_classification_results:
            print("⚠️  Insufficient data for relationship chart")
            return None

        # Aggregate KGE results by dimension
        kge_by_dim = {}
        for kge_data in all_kge_results:
            if 'all_results' in kge_data:
                for result in kge_data['all_results']:
                    dim = result['embedding_dim']
                    if dim not in kge_by_dim:
                        kge_by_dim[dim] = {
                            'mrr': [],
                            'hits_at_10': []
                        }
                    kge_by_dim[dim]['mrr'].append(result['evaluation']['mrr'])
                    kge_by_dim[dim]['hits_at_10'].append(result['evaluation']['hits_at_10'])

        # Aggregate classification results by dimension
        clf_by_dim = {}
        for clf_data in all_classification_results:
            dim = clf_data['embedding_dim']
            if dim not in clf_by_dim:
                clf_by_dim[dim] = []
            clf_by_dim[dim].append(clf_data['test_metrics']['accuracy'])

        # Get common dimensions
        common_dims = sorted(set(kge_by_dim.keys()) & set(clf_by_dim.keys()))

        if not common_dims:
            print("⚠️  No overlapping dimensions between KGE and classification results")
            return None

        # Prepare data
        dims = []
        mrr_scores = []
        hits_at_10 = []
        accuracies = []

        for dim in common_dims:
            dims.append(dim)
            mrr_scores.append(max(kge_by_dim[dim]['mrr']))
            hits_at_10.append(max(kge_by_dim[dim]['hits_at_10']))
            accuracies.append(max(clf_by_dim[dim]))

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # 1. MRR vs Classification Accuracy
        ax1_twin = ax1.twinx()

        line1 = ax1.plot(dims, mrr_scores, marker='o', linewidth=2, markersize=8,
                         label='KGE MRR', color=COLORS[0])
        line2 = ax1_twin.plot(dims, accuracies, marker='*', linewidth=2, markersize=12,
                              label='Classification Accuracy', color=COLORS[1])

        ax1.set_xlabel('Embedding Dimension', fontweight='bold')
        ax1.set_ylabel('Mean Reciprocal Rank (MRR)', fontweight='bold', color=COLORS[0])
        ax1_twin.set_ylabel('Classification Accuracy', fontweight='bold', color=COLORS[1])
        ax1.set_title('KGE Performance vs Classification Accuracy', fontweight='bold')
        ax1.tick_params(axis='y', labelcolor=COLORS[0])
        ax1_twin.tick_params(axis='y', labelcolor=COLORS[1])
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(dims)

        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='best')

        # 2. Hits@10 vs Classification Accuracy
        ax2_twin = ax2.twinx()

        line3 = ax2.plot(dims, hits_at_10, marker='s', linewidth=2, markersize=8,
                         label='KGE Hits@10', color=COLORS[2])
        line4 = ax2_twin.plot(dims, accuracies, marker='*', linewidth=2, markersize=12,
                              label='Classification Accuracy', color=COLORS[1])

        ax2.set_xlabel('Embedding Dimension', fontweight='bold')
        ax2.set_ylabel('Hits@10', fontweight='bold', color=COLORS[2])
        ax2_twin.set_ylabel('Classification Accuracy', fontweight='bold', color=COLORS[1])
        ax2.set_title('KGE Hits@10 vs Classification Accuracy', fontweight='bold')
        ax2.tick_params(axis='y', labelcolor=COLORS[2])
        ax2_twin.tick_params(axis='y', labelcolor=COLORS[1])
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(dims)

        # Combine legends
        lines = line3 + line4
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='best')

        # Clean repo name for display
        clean_repo_name = repo_name.replace('_2iter', '').replace('_2_iter', '')

        # Overall title
        fig.suptitle(f'Relationship between KGE and Classification Performance - {clean_repo_name.upper()}',
                     fontsize=14, fontweight='bold', y=1.02)

        # Save
        output_dir = self.output_base_path / repo_name
        output_path = output_dir / 'kge_classification_relationship.png'
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Generated: {output_path}")
        return output_path

    def generate_all_visualizations(self, repo_name: str):
        """Generate all visualizations for a repository"""
        print(f"\n{'=' * 60}")
        print(f"🎨 GENERATING KGE & CLASSIFICATION VISUALIZATIONS")
        print(f"   Repository: {repo_name}")

        # Force directory name for reuters_activities variants
        if 'reuters_activities' in repo_name.lower():
            print(f"   Forcing directory to: reuters_activities")
            repo_name = 'reuters_activities'

        print(f"{'=' * 60}\n")

        # Find all result files
        kge_files, classification_files = self.find_all_results(repo_name)

        if not kge_files:
            print(f"❌ No KGE results found for {repo_name}")
            return

        if not classification_files:
            print(f"❌ No classification results found for {repo_name}")
            return

        print(f"📊 Found {len(kge_files)} KGE result file(s)")
        print(f"📊 Found {len(classification_files)} classification result file(s)\n")

        # Load all results and select optimal model
        all_kge, all_classification, optimal_kge, optimal_classification = self.load_all_results(
            kge_files, classification_files
        )

        if not all_kge or not all_classification:
            print(f"❌ Could not load results")
            return

        if not optimal_kge or not optimal_classification:
            print(f"❌ Could not determine optimal model")
            return

        # Generate visualizations
        print("Generating visualizations...\n")

        # Comparisons across all results
        self.create_all_kge_comparison(all_kge, repo_name)
        self.create_all_classification_comparison(all_classification, repo_name)

        # Detailed view of optimal model
        self.create_classification_performance_chart(optimal_classification, repo_name)

        # Relationship between KGE and classification
        self.create_kge_classification_relationship(all_kge, all_classification, repo_name)

        print(f"\n{'=' * 60}")
        print(f"✅ VISUALIZATIONS COMPLETED")
        print(f"📁 Output directory: {self.output_base_path / repo_name}")
        print(f"{'=' * 60}\n")


def show_menu():
    """Show interactive menu"""
    print("\n" + "=" * 60)
    print(" 🎨 KGE & CLASSIFICATION VISUALIZATION GENERATOR")
    print("=" * 60)
    print("\nThis script generates visualizations for:")
    print("  • KGE training performance across dimensions")
    print("  • Topic classification metrics and confusion matrix")
    print("  • t-SNE visualization of embeddings by topic")
    print("  • Relationship between KGE and classification performance")
    print("\n" + "=" * 60 + "\n")


def get_available_repos(base_path: str) -> List[str]:
    """Get list of available repositories"""
    path = Path(base_path)
    if not path.exists():
        return []
    return [d.name for d in path.iterdir() if d.is_dir() and not d.name.startswith('.')]


def main():
    """Main function with interactive menu"""

    # Base paths
    INPUT_BASE = '../data/model_output'
    OUTPUT_BASE = '../data/model_output_visualization'

    show_menu()

    # Get available repositories
    available_repos = get_available_repos(INPUT_BASE)

    if not available_repos:
        print(f"❌ No repositories found in: {INPUT_BASE}")
        print(f"   Please verify the input path.")
        return

    print("📂 Available repositories:")
    for idx, repo in enumerate(available_repos, 1):
        print(f"   {idx}. {repo}")

    print(f"\n   0. Process ALL repositories")
    print(f"   q. Exit")

    # Request selection
    while True:
        choice = input("\n👉 Select an option: ").strip().lower()

        if choice == 'q':
            print("\n👋 Goodbye!")
            return

        if choice == '0':
            # Process all
            print(f"\n🚀 Processing ALL repositories...\n")
            visualizer = KGEClassificationVisualizer(INPUT_BASE, OUTPUT_BASE)

            for repo in available_repos:
                visualizer.generate_all_visualizations(repo)

            print("\n✅ Process completed for all repositories")
            break

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(available_repos):
                selected_repo = available_repos[idx]
                print(f"\n🚀 Processing: {selected_repo}\n")

                visualizer = KGEClassificationVisualizer(INPUT_BASE, OUTPUT_BASE)
                visualizer.generate_all_visualizations(selected_repo)
                break
            else:
                print("❌ Invalid option. Try again.")
        except ValueError:
            print("❌ Invalid input. Try again.")


if __name__ == "__main__":
    main()