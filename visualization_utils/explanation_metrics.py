#!/usr/bin/env python3
"""
Visualization Generator for Topic Explanation Analysis
Generates representative charts from explanation summary data
"""

import json
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional
import pandas as pd
from matplotlib.patches import Rectangle
import warnings

warnings.filterwarnings('ignore')

# Configuración de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']


class VisualizationGenerator:
    """Genera visualizaciones para análisis de explicaciones de topics"""

    def __init__(self, input_base_path: str, output_base_path: str):
        self.input_base_path = Path(input_base_path)
        self.output_base_path = Path(output_base_path)

    def load_summary_data(self, repo_name: str) -> Optional[Dict]:
        """Carga el resumen global de todos los topics"""
        summary_path = self.input_base_path / repo_name / 'all_topics_summary.json'

        if not summary_path.exists():
            print(f"❌ No se encontró el archivo: {summary_path}")
            return None

        with open(summary_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_topic_details(self, repo_name: str, topic_id: int) -> Optional[Dict]:
        """Carga los detalles de un topic específico"""
        detail_paths = [
            self.input_base_path / repo_name / f'topic_{topic_id}' / 'evaluations.json',
            self.input_base_path / repo_name / f'topic_{topic_id}' / 'explanations.json',
            self.input_base_path / repo_name / f'topic_{topic_id}' / 'clusters.json'
        ]

        data = {}
        for path in detail_paths:
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    key = path.stem
                    data[key] = json.load(f)

        return data if data else None

    def create_metrics_radar_chart(self, summary_data: Dict, repo_name: str):
        """
        Crea un radar chart comparando métricas promedio por topic
        """
        topics_metrics = summary_data['topics_metrics']

        # Preparar datos
        categories = ['Coherence', 'Relevance', 'Coverage']
        num_topics = len(topics_metrics)

        # Configurar el radar chart
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

        # Plotear cada topic
        for idx, topic in enumerate(topics_metrics):
            values = [
                topic['avg_coherence'],
                topic['avg_relevance'],
                topic['avg_coverage']
            ]
            values += values[:1]

            color = COLORS[idx % len(COLORS)]
            ax.plot(angles, values, 'o-', linewidth=2, label=f"Topic {topic['topic_id']}", color=color)
            ax.fill(angles, values, alpha=0.15, color=color)

        # Configuración
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=11, weight='bold')
        ax.set_ylim(0, 5)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_yticklabels(['1', '2', '3', '4', '5'], size=9)
        ax.grid(True, linestyle='--', alpha=0.7)

        plt.title(f'Metrics Comparison by Topic\nRepository: {repo_name}',
                  size=14, weight='bold', pad=20)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)

        # Guardar
        output_dir = self.output_base_path / repo_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / 'metrics_radar_comparison.png'
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Generado: {output_path}")
        return output_path

    def create_metrics_heatmap(self, summary_data: Dict, repo_name: str):
        """
        Crea un heatmap de métricas por topic y número de clusters
        """
        topics_metrics = summary_data['topics_metrics']

        # Preparar datos para el heatmap
        data_matrix = []
        topic_labels = []

        for topic in topics_metrics:
            data_matrix.append([
                topic['avg_coherence'],
                topic['avg_relevance'],
                topic['avg_coverage'],
                topic['num_clusters']
            ])
            topic_labels.append(f"Topic {topic['topic_id']}")

        df = pd.DataFrame(
            data_matrix,
            index=topic_labels,
            columns=['Coherence', 'Relevance', 'Coverage', 'Sub-domains Identified as Clusters']
        )

        # Crear figura
        fig, ax = plt.subplots(figsize=(12, 6))

        # Normalizar la columna de clusters para la visualización
        df_normalized = df.copy()
        df_normalized['Sub-domains Identified as Clusters'] = df_normalized['Sub-domains Identified as Clusters'] / \
                                                              df_normalized[
                                                                  'Sub-domains Identified as Clusters'].max() * 5

        # Crear heatmap
        sns.heatmap(df_normalized, annot=df.values, fmt='.2f', cmap='RdYlGn',
                    cbar_kws={'label': 'Score (normalized)'},
                    linewidths=0.5, linecolor='gray', ax=ax,
                    vmin=0, vmax=5)

        plt.title(f'Metrics Heatmap by Topic\nRepository: {repo_name}',
                  size=14, weight='bold', pad=15)
        plt.xlabel('Metrics', size=12, weight='bold')
        plt.ylabel('Topics', size=12, weight='bold')

        # Guardar
        output_dir = self.output_base_path / repo_name
        output_path = output_dir / 'metrics_heatmap.png'
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Generado: {output_path}")
        return output_path

    def create_cluster_quality_chart(self, repo_name: str, summary_data: Dict):
        """
        Crea un gráfico de barras para la calidad de clustering (Silhouette) por topic
        """
        # Cargar datos de silhouette de cada topic
        topics_metrics = summary_data['topics_metrics']

        silhouette_scores = []
        topic_ids = []

        for topic in topics_metrics:
            topic_id = topic['topic_id']
            # Intentar cargar config para obtener silhouette
            config_path = self.input_base_path / repo_name / f'topic_{topic_id}' / f'config_topic_{topic_id}.json'

            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    silhouette = config.get('silhouette_score', 0)
                    silhouette_scores.append(silhouette)
                    topic_ids.append(f"Topic {topic_id}")

        if not silhouette_scores:
            print("⚠️  No se encontraron scores de silhouette")
            return None

        # Crear gráfico
        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.bar(topic_ids, silhouette_scores, color=COLORS[:len(topic_ids)],
                      edgecolor='black', linewidth=1.5, alpha=0.8)

       # # Añadir líneas de referencia
       # ax.axhline(y=0.5, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Good (>0.5)')
       # ax.axhline(y=0.25, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='Fair (>0.25)')

        # Añadir valores en las barras
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=11, weight='bold')

        # Configuración
        ax.set_ylabel('Silhouette Score', size=12, weight='bold')
        ax.set_xlabel('Topics', size=12, weight='bold')
        ax.set_title(f'Clustering Quality by Topic\nRepository: {repo_name}',
                     size=14, weight='bold', pad=15)
        ax.set_ylim(0, max(silhouette_scores) * 1.2)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Guardar
        output_dir = self.output_base_path / repo_name
        output_path = output_dir / 'clustering_quality.png'
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Generado: {output_path}")
        return output_path

    def create_average_score_chart(self, summary_data: Dict, repo_name: str):
        """
        Crea un gráfico de barras con el score promedio global por topic
        """
        topics_metrics = summary_data['topics_metrics']

        # Calcular score promedio por topic
        topic_labels = []
        avg_scores = []

        for topic in topics_metrics:
            avg_score = (topic['avg_coherence'] + topic['avg_relevance'] + topic['avg_coverage']) / 3
            topic_labels.append(f"Topic {topic['topic_id']}")
            avg_scores.append(avg_score)

        # Crear gráfico
        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.bar(topic_labels, avg_scores, color=COLORS[:len(topic_labels)],
                      edgecolor='black', linewidth=1.5, alpha=0.8)

        # Añadir líneas de referencia
       # ax.axhline(y=4.0, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Excellent (≥4.0)')
       # ax.axhline(y=3.0, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='Good (≥3.0)')
       # ax.axhline(y=2.0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Fair (≥2.0)')

        # Añadir valores en las barras
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=12, weight='bold')

        # Configuración
        ax.set_ylabel('Average Score', size=12, weight='bold')
        ax.set_xlabel('Topics', size=12, weight='bold')
        ax.set_title(f'Average Score by Topic\nRepository: {repo_name}',
                     size=14, weight='bold', pad=15)
        ax.set_ylim(0, 5)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Guardar
        output_dir = self.output_base_path / repo_name
        output_path = output_dir / 'average_score_by_topic.png'
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Generado: {output_path}")
        return output_path

    def create_comprehensive_dashboard(self, repo_name: str, summary_data: Dict):
        """
        Crea un dashboard completo con múltiples visualizaciones
        """
        topics_metrics = summary_data['topics_metrics']

        # Crear figura con subplots
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # 1. Comparación de métricas por topic (barras agrupadas)
        ax1 = fig.add_subplot(gs[0, 0])
        metrics_df = pd.DataFrame([
            {
                'Topic': f"T{t['topic_id']}",
                'Coherence': t['avg_coherence'],
                'Relevance': t['avg_relevance'],
                'Coverage': t['avg_coverage']
            }
            for t in topics_metrics
        ])

        x = np.arange(len(metrics_df))
        width = 0.25

        ax1.bar(x - width, metrics_df['Coherence'], width, label='Coherence', color=COLORS[0], alpha=0.8)
        ax1.bar(x, metrics_df['Relevance'], width, label='Relevance', color=COLORS[1], alpha=0.8)
        ax1.bar(x + width, metrics_df['Coverage'], width, label='Coverage', color=COLORS[2], alpha=0.8)

        ax1.set_xlabel('Topics', fontweight='bold')
        ax1.set_ylabel('Score', fontweight='bold')
        ax1.set_title('Metrics by Topic', fontweight='bold', fontsize=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics_df['Topic'])
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim(0, 5)

        # 2. Distribución de clusters
        ax2 = fig.add_subplot(gs[0, 1])
        cluster_counts = [t['num_clusters'] for t in topics_metrics]
        topic_labels = [f"Topic {t['topic_id']}" for t in topics_metrics]

        wedges, texts, autotexts = ax2.pie(cluster_counts, labels=topic_labels, autopct='%1.0f%%',
                                           colors=COLORS[:len(cluster_counts)], startangle=90)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        ax2.set_title('Cluster Distribution', fontweight='bold', fontsize=12)

        # 3. Average Score by Topic
        ax3 = fig.add_subplot(gs[1, 0])

        topic_labels_short = [f"T{t['topic_id']}" for t in topics_metrics]
        avg_scores = [(t['avg_coherence'] + t['avg_relevance'] + t['avg_coverage']) / 3
                      for t in topics_metrics]

        bars = ax3.bar(topic_labels_short, avg_scores, color=COLORS[:len(topic_labels_short)],
                       edgecolor='black', linewidth=1.2, alpha=0.8)

        # Añadir valores en las barras
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.2f}',
                     ha='center', va='bottom', fontsize=10, weight='bold')

        # Líneas de referencia
        #ax3.axhline(y=4.0, color='green', linestyle='--', linewidth=1.5, alpha=0.4)
      #  ax3.axhline(y=3.0, color='orange', linestyle='--', linewidth=1.5, alpha=0.4)

        ax3.set_xlabel('Topics', fontweight='bold')
        ax3.set_ylabel('Average Score', fontweight='bold')
        ax3.set_title('Average Score by Topic', fontweight='bold', fontsize=12)
        ax3.grid(axis='y', alpha=0.3)
        ax3.set_ylim(0, 5)

        # 4. Resumen estadístico
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')

        # Calcular estadísticas globales
        avg_coherence = np.mean([t['avg_coherence'] for t in topics_metrics])
        avg_relevance = np.mean([t['avg_relevance'] for t in topics_metrics])
        avg_coverage = np.mean([t['avg_coverage'] for t in topics_metrics])
        total_clusters = sum([t['num_clusters'] for t in topics_metrics])

        summary_text = f"""
        GLOBAL SUMMARY
        {'=' * 40}

        Repository: {repo_name}
        Total Topics: {summary_data['total_topics']}
        Total Clusters: {total_clusters}

        AVERAGE METRICS
        {'─' * 40}
        Coherence:  {avg_coherence:.2f}/5.00
        Relevance:  {avg_relevance:.2f}/5.00
        Coverage:   {avg_coverage:.2f}/5.00

        OVERALL QUALITY
        {'─' * 40}
        Global Score: {(avg_coherence + avg_relevance + avg_coverage) / 3:.2f}/5.00
        """

        ax4.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                 verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        # Título general
        fig.suptitle(f'Analysis Dashboard - {repo_name}',
                     fontsize=16, fontweight='bold', y=0.98)

        # Guardar
        output_dir = self.output_base_path / repo_name
        output_path = output_dir / 'comprehensive_dashboard.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Generado: {output_path}")
        return output_path

    def generate_all_visualizations(self, repo_name: str):
        """Genera todas las visualizaciones para un repositorio"""
        print(f"\n{'=' * 60}")
        print(f"🎨 GENERANDO VISUALIZACIONES PARA: {repo_name}")
        print(f"{'=' * 60}\n")

        # Cargar datos
        summary_data = self.load_summary_data(repo_name)
        if not summary_data:
            print(f"❌ No se pudo cargar el resumen para {repo_name}")
            return

        print(f"📊 Repositorio: {summary_data['repository']}")
        print(f"📈 Topics procesados: {summary_data['topics_processed']}/{summary_data['total_topics']}\n")

        # Generar visualizaciones
        print("Generando visualizaciones...\n")

        self.create_metrics_radar_chart(summary_data, repo_name)
        self.create_metrics_heatmap(summary_data, repo_name)
        self.create_cluster_quality_chart(repo_name, summary_data)
        self.create_average_score_chart(summary_data, repo_name)
        self.create_comprehensive_dashboard(repo_name, summary_data)

        print(f"\n{'=' * 60}")
        print(f"✅ VISUALIZACIONES COMPLETADAS")
        print(f"📁 Directorio de salida: {self.output_base_path / repo_name}")
        print(f"{'=' * 60}\n")


def show_menu():
    """Muestra el menú interactivo"""
    print("\n" + "=" * 60)
    print(" 🎨 GENERADOR DE VISUALIZACIONES - ANÁLISIS DE TOPICS")
    print("=" * 60)
    print("\nEste script genera visualizaciones para el análisis de")
    print("explicaciones de topics y clusters.\n")
    print("Visualizaciones generadas:")
    print("  • Radar chart de métricas por topic")
    print("  • Heatmap de métricas")
    print("  • Gráfico de calidad de clustering")
    print("  • Dashboard comprehensivo")
    print("\n" + "=" * 60 + "\n")


def get_available_repos(base_path: str) -> List[str]:
    """Obtiene la lista de repositorios disponibles"""
    path = Path(base_path)
    if not path.exists():
        return []
    return [d.name for d in path.iterdir() if d.is_dir() and not d.name.startswith('.')]


def main():
    """Función principal con menú interactivo"""

    # Rutas base
    INPUT_BASE = '../data/explanations_eng'
    OUTPUT_BASE = '../data/explanations_eng_visualization'

    show_menu()

    # Obtener repositorios disponibles
    available_repos = get_available_repos(INPUT_BASE)

    if not available_repos:
        print(f"❌ No se encontraron repositorios en: {INPUT_BASE}")
        print(f"   Por favor, verifica la ruta de entrada.")
        return

    print("📂 Repositorios disponibles:")
    for idx, repo in enumerate(available_repos, 1):
        print(f"   {idx}. {repo}")

    print(f"\n   0. Procesar TODOS los repositorios")
    print(f"   q. Salir")

    # Solicitar selección
    while True:
        choice = input("\n👉 Selecciona una opción: ").strip().lower()

        if choice == 'q':
            print("\n👋 ¡Hasta luego!")
            return

        if choice == '0':
            # Procesar todos
            print(f"\n🚀 Procesando TODOS los repositorios...\n")
            generator = VisualizationGenerator(INPUT_BASE, OUTPUT_BASE)

            for repo in available_repos:
                generator.generate_all_visualizations(repo)

            print("\n✅ Proceso completado para todos los repositorios")
            break

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(available_repos):
                selected_repo = available_repos[idx]
                print(f"\n🚀 Procesando: {selected_repo}\n")

                generator = VisualizationGenerator(INPUT_BASE, OUTPUT_BASE)
                generator.generate_all_visualizations(selected_repo)
                break
            else:
                print("❌ Opción inválida. Intenta de nuevo.")
        except ValueError:
            print("❌ Entrada inválida. Intenta de nuevo.")


if __name__ == "__main__":
    main()