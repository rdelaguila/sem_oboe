# File: compare_explanations_methods.py
"""
Comparative visualization of explanation results between Hierarchical Clustering and spectral
Reads and compares all_topics_summary.json files from both approaches
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ExplanationComparator:
    def __init__(self, repo_name: str):
        """
        Initialize comparator for a specific repository

        Args:
            repo_name: Name of the repository (amazon, bbc, reuters)
        """
        if repo_name == 'reuters':
            repo_name = 'reuters_activities'
        self.repo_name = repo_name
        self.hierarchical_path = f'../data/explanations_eng_jerarquico/{repo_name}/all_topics_summary.json'
        self.spectral_path = f'../data/explanations_eng_spectral/{repo_name}/all_topics_summary.json'

        self.hierarchical_data = None
        self.spectral_data = None
        self.comparison_df = None

    def load_data(self):
        """
        Load summary data from both methods
        """
        # Load hierarchical clustering results
        if os.path.exists(self.hierarchical_path):
            with open(self.hierarchical_path, 'r') as f:
                self.hierarchical_data = json.load(f)
            print(f"✅ Loaded hierarchical clustering results for {self.repo_name}")
        else:
            print(f"❌ Hierarchical clustering results not found at {self.hierarchical_path}")
            return False

        # Load spectral results
        if os.path.exists(self.spectral_path):
            with open(self.spectral_path, 'r') as f:
                self.spectral_data = json.load(f)
            print(f"✅ Loaded spectral results for {self.repo_name}")
        else:
            print(f"❌ spectral results not found at {self.spectral_path}")
            return False

        # Create comparison dataframe
        self._create_comparison_df()
        return True

    def _create_comparison_df(self):
        """
        Create a unified dataframe for comparison
        """
        data = []

        # Process hierarchical data
        for topic in self.hierarchical_data['topics_metrics']:
            data.append({
                'topic_id': topic['topic_id'],
                'method': 'Hierarchical',
                'num_clusters': topic['num_clusters'],
                'coherence': topic['avg_coherence'],
                'relevance': topic['avg_relevance'],
                'coverage': topic['avg_coverage']
            })

        # Process spectral data
        for topic in self.spectral_data['topics_metrics']:
            data.append({
                'topic_id': topic['topic_id'],
                'method': 'spectral',
                'num_clusters': topic['num_clusters'],
                'coherence': topic['avg_coherence'],
                'relevance': topic['avg_relevance'],
                'coverage': topic['avg_coverage']
            })

        self.comparison_df = pd.DataFrame(data)

    def plot_metric_comparison_bars(self, save_path: str = None):
        """
        Create grouped bar plots comparing metrics between methods
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Comparison of Methods: {self.repo_name.upper()}', fontsize=16, fontweight='bold')

        metrics = ['coherence', 'relevance', 'coverage', 'num_clusters']
        titles = ['Coherence Scores', 'Relevance Scores', 'Coverage Scores', 'Number of Clusters']

        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx // 2, idx % 2]

            # Prepare data for grouped bars
            hierarchical_data = self.comparison_df[self.comparison_df['method'] == 'Hierarchical'].sort_values(
                'topic_id')
            spectral_data = self.comparison_df[self.comparison_df['method'] == 'spectral'].sort_values('topic_id')

            x = np.arange(len(hierarchical_data))
            width = 0.35

            bars1 = ax.bar(x - width / 2, hierarchical_data[metric], width,
                           label='Hierarchical', alpha=0.8, color='steelblue')
            bars2 = ax.bar(x + width / 2, spectral_data[metric], width,
                           label='spectral', alpha=0.8, color='coral')

            ax.set_xlabel('Topic ID')
            ax.set_ylabel('Score' if metric != 'num_clusters' else 'Number')
            ax.set_title(title)
            ax.set_xticks(x)
            ax.set_xticklabels(hierarchical_data['topic_id'].values)
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Add value labels on bars
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2., height,
                            f'{height:.2f}' if metric != 'num_clusters' else f'{int(height)}',
                            ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Bar comparison plot saved to {save_path}")

        plt.show()

    def plot_radar_comparison(self, save_path: str = None):
        """
        Create radar charts comparing overall performance
        """
        # Calculate average metrics for each method
        avg_metrics = self.comparison_df.groupby('method')[['coherence', 'relevance', 'coverage']].mean()

        fig = go.Figure()

        categories = ['Coherence', 'Relevance', 'Coverage']

        # Add trace for Hierarchical
        fig.add_trace(go.Scatterpolar(
            r=avg_metrics.loc['Hierarchical'].values,
            theta=categories,
            fill='toself',
            name='Hierarchical',
            line_color='blue',
            fillcolor='rgba(0, 0, 255, 0.25)'
        ))

        # Add trace for spectral
        fig.add_trace(go.Scatterpolar(
            r=avg_metrics.loc['spectral'].values,
            theta=categories,
            fill='toself',
            name='spectral',
            line_color='red',
            fillcolor='rgba(255, 0, 0, 0.25)'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 5]
                )
            ),
            showlegend=True,
            title=f"Overall Performance Comparison - {self.repo_name.upper()}"
        )

        if save_path:
            fig.write_html(save_path.replace('.png', '.html'))
            print(f"✅ Radar plot saved to {save_path.replace('.png', '.html')}")

        fig.show()

    def plot_performance_heatmap(self, save_path: str = None):
        """
        Create heatmap showing performance differences
        """
        # Pivot data for heatmap
        hierarchical_pivot = self.comparison_df[self.comparison_df['method'] == 'Hierarchical'].set_index('topic_id')[
            ['coherence', 'relevance', 'coverage']]
        spectral_pivot = self.comparison_df[self.comparison_df['method'] == 'spectral'].set_index('topic_id')[
            ['coherence', 'relevance', 'coverage']]

        # Calculate difference (spectral - Hierarchical)
        diff_pivot = spectral_pivot - hierarchical_pivot

        fig, ax = plt.subplots(figsize=(10, 6))

        sns.heatmap(diff_pivot.T, annot=True, fmt='.2f', cmap='RdBu_r',
                    center=0, cbar_kws={'label': 'Difference (spectral - Hierarchical)'},
                    ax=ax, vmin=-1, vmax=1)

        ax.set_title(f'Performance Difference Heatmap - {self.repo_name.upper()}')
        ax.set_xlabel('Topic ID')
        ax.set_ylabel('Metric')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Heatmap saved to {save_path}")

        plt.show()

    def plot_topic_by_topic_comparison(self, save_path: str = None):
        """
        Create line plots showing topic-by-topic comparison
        """
        fig = make_subplots(rows=2, cols=2,
                            subplot_titles=('Coherence', 'Relevance', 'Coverage', 'Number of Clusters'))

        metrics = ['coherence', 'relevance', 'coverage', 'num_clusters']

        # Get unique topics sorted
        topics = sorted(self.comparison_df['topic_id'].unique())

        for idx, metric in enumerate(metrics):
            row = idx // 2 + 1
            col = idx % 2 + 1

            hierarchical_values = []
            spectral_values = []

            for topic in topics:
                hier_val = self.comparison_df[(self.comparison_df['topic_id'] == topic) &
                                              (self.comparison_df['method'] == 'Hierarchical')][metric].values[0]
                bert_val = self.comparison_df[(self.comparison_df['topic_id'] == topic) &
                                              (self.comparison_df['method'] == 'spectral')][metric].values[0]
                hierarchical_values.append(hier_val)
                spectral_values.append(bert_val)

            fig.add_trace(
                go.Scatter(x=topics, y=hierarchical_values, name='Hierarchical',
                           line=dict(color='blue', width=2), mode='lines+markers',
                           showlegend=(idx == 0)),
                row=row, col=col
            )

            fig.add_trace(
                go.Scatter(x=topics, y=spectral_values, name='spectral',
                           line=dict(color='red', width=2), mode='lines+markers',
                           showlegend=(idx == 0)),
                row=row, col=col
            )

        fig.update_xaxes(title_text="Topic ID")
        fig.update_yaxes(title_text="Score", row=1, col=1)
        fig.update_yaxes(title_text="Score", row=1, col=2)
        fig.update_yaxes(title_text="Score", row=2, col=1)
        fig.update_yaxes(title_text="Number", row=2, col=2)

        fig.update_layout(height=800, title_text=f"Topic-by-Topic Comparison - {self.repo_name.upper()}")

        if save_path:
            fig.write_html(save_path.replace('.png', '.html'))
            print(f"✅ Line comparison plot saved to {save_path.replace('.png', '.html')}")

        fig.show()

    def generate_statistical_summary(self, save_path: str = None):
        """
        Generate statistical summary comparing both methods
        """
        summary = {}

        for method in ['Hierarchical', 'spectral']:
            method_data = self.comparison_df[self.comparison_df['method'] == method]

            summary[method] = {
                'coherence': {
                    'mean': method_data['coherence'].mean(),
                    'std': method_data['coherence'].std(),
                    'min': method_data['coherence'].min(),
                    'max': method_data['coherence'].max()
                },
                'relevance': {
                    'mean': method_data['relevance'].mean(),
                    'std': method_data['relevance'].std(),
                    'min': method_data['relevance'].min(),
                    'max': method_data['relevance'].max()
                },
                'coverage': {
                    'mean': method_data['coverage'].mean(),
                    'std': method_data['coverage'].std(),
                    'min': method_data['coverage'].min(),
                    'max': method_data['coverage'].max()
                },
                'num_clusters': {
                    'mean': method_data['num_clusters'].mean(),
                    'std': method_data['num_clusters'].std(),
                    'min': method_data['num_clusters'].min(),
                    'max': method_data['num_clusters'].max(),
                    'total': method_data['num_clusters'].sum()
                }
            }

        # Calculate improvements
        improvements = {}
        for metric in ['coherence', 'relevance', 'coverage']:
            improvements[metric] = {
                'absolute': summary['spectral'][metric]['mean'] - summary['Hierarchical'][metric]['mean'],
                'relative': ((summary['spectral'][metric]['mean'] - summary['Hierarchical'][metric]['mean']) /
                             summary['Hierarchical'][metric]['mean'] * 100) if summary['Hierarchical'][metric][
                                                                                   'mean'] != 0 else 0
            }

        # Generate report
        report = f"""
STATISTICAL COMPARISON REPORT
Repository: {self.repo_name.upper()}
===============================================

1. OVERALL PERFORMANCE METRICS
-------------------------------
                  HIERARCHICAL        spectral
Coherence:        {summary['Hierarchical']['coherence']['mean']:.3f} ± {summary['Hierarchical']['coherence']['std']:.3f}    {summary['spectral']['coherence']['mean']:.3f} ± {summary['spectral']['coherence']['std']:.3f}
Relevance:        {summary['Hierarchical']['relevance']['mean']:.3f} ± {summary['Hierarchical']['relevance']['std']:.3f}    {summary['spectral']['relevance']['mean']:.3f} ± {summary['spectral']['relevance']['std']:.3f}
Coverage:         {summary['Hierarchical']['coverage']['mean']:.3f} ± {summary['Hierarchical']['coverage']['std']:.3f}    {summary['spectral']['coverage']['mean']:.3f} ± {summary['spectral']['coverage']['std']:.3f}

2. CLUSTERING STATISTICS
-------------------------
                  HIERARCHICAL        spectral
Avg Clusters:     {summary['Hierarchical']['num_clusters']['mean']:.1f}              {summary['spectral']['num_clusters']['mean']:.1f}
Total Clusters:   {int(summary['Hierarchical']['num_clusters']['total'])}                {int(summary['spectral']['num_clusters']['total'])}

3. PERFORMANCE DIFFERENCES (spectral vs Hierarchical)
------------------------------------------------------
Coherence:  {improvements['coherence']['absolute']:+.3f} ({improvements['coherence']['relative']:+.1f}%)
Relevance:  {improvements['relevance']['absolute']:+.3f} ({improvements['relevance']['relative']:+.1f}%)
Coverage:   {improvements['coverage']['absolute']:+.3f} ({improvements['coverage']['relative']:+.1f}%)

4. TOPIC-LEVEL ANALYSIS
------------------------
Total topics analyzed: {len(self.comparison_df['topic_id'].unique())}

Best performing topics (spectral):
"""

        # Add best performing topics
        spectral_data = self.comparison_df[self.comparison_df['method'] == 'spectral']
        overall_score = (spectral_data['coherence'] + spectral_data['relevance'] + spectral_data['coverage']) / 3
        spectral_data_with_score = spectral_data.copy()
        spectral_data_with_score['overall_score'] = overall_score

        top_topics = spectral_data_with_score.nlargest(3, 'overall_score')
        for _, row in top_topics.iterrows():
            report += f"  Topic {row['topic_id']}: Overall Score = {row['overall_score']:.3f}\n"

        report += """
5. RECOMMENDATION
-----------------
"""

        # Determine which method performs better
        spectral_overall = (summary['spectral']['coherence']['mean'] +
                            summary['spectral']['relevance']['mean'] +
                            summary['spectral']['coverage']['mean']) / 3

        hierarchical_overall = (summary['Hierarchical']['coherence']['mean'] +
                                summary['Hierarchical']['relevance']['mean'] +
                                summary['Hierarchical']['coverage']['mean']) / 3

        if spectral_overall > hierarchical_overall:
            report += f"spectral shows better overall performance ({spectral_overall:.3f} vs {hierarchical_overall:.3f})"
        else:
            report += f"Hierarchical clustering shows better overall performance ({hierarchical_overall:.3f} vs {spectral_overall:.3f})"

        print(report)

        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"\n✅ Statistical summary saved to {save_path}")

        return summary, improvements

    def generate_all_visualizations(self, output_dir: str = None):
        """
        Generate all visualizations and save them
        """
        if output_dir is None:
            output_dir = f'../visualizations/comparison_{self.repo_name}'

        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"GENERATING COMPARISON VISUALIZATIONS FOR {self.repo_name.upper()}")
        print(f"{'=' * 60}")

        # 1. Bar comparison
        print("\n1. Generating bar comparison plots...")
        self.plot_metric_comparison_bars(
            save_path=os.path.join(output_dir, 'metric_comparison_bars.png')
        )

        # 2. Radar chart
        print("\n2. Generating radar chart...")
        self.plot_radar_comparison(
            save_path=os.path.join(output_dir, 'radar_comparison.png')
        )

        # 3. Heatmap
        print("\n3. Generating performance difference heatmap...")
        self.plot_performance_heatmap(
            save_path=os.path.join(output_dir, 'performance_heatmap.png')
        )

        # 4. Line plots
        print("\n4. Generating topic-by-topic comparison...")
        self.plot_topic_by_topic_comparison(
            save_path=os.path.join(output_dir, 'topic_comparison_lines.png')
        )

        # 5. Statistical summary
        print("\n5. Generating statistical summary...")
        self.generate_statistical_summary(
            save_path=os.path.join(output_dir, 'statistical_summary.txt')
        )

        print(f"\n✅ All visualizations saved to {output_dir}")


def main():
    """
    Main function to run comparisons
    """
    print("EXPLANATION METHOD COMPARISON TOOL")
    print("=" * 50)

    # Available repositories
    repositories = ['amazon', 'bbc', 'reuters']

    print("\nAvailable repositories:")
    for idx, repo in enumerate(repositories, 1):
        print(f"{idx}. {repo}")
    print(f"{len(repositories) + 1}. All repositories")

    try:
        choice = int(input("\nSelect option (number): "))

        if choice == len(repositories) + 1:
            # Process all repositories
            for repo in repositories:
                print(f"\n{'=' * 60}")
                print(f"Processing {repo.upper()}")
                print(f"{'=' * 60}")

                comparator = ExplanationComparator(repo)
                if comparator.load_data():
                    comparator.generate_all_visualizations()
                else:
                    print(f"⚠️ Skipping {repo} due to missing data")

        elif 1 <= choice <= len(repositories):
            # Process single repository
            repo = repositories[choice - 1]
            comparator = ExplanationComparator(repo)

            if comparator.load_data():
                comparator.generate_all_visualizations()
            else:
                print("❌ Cannot proceed without data from both methods")

        else:
            print("❌ Invalid selection")

    except ValueError:
        print("❌ Invalid input")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()