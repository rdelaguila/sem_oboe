#!/usr/bin/env python3
"""
Component C: Integrated KGE + Classification Pipeline with Guan et al. (2025) CES Strategy
==============================================================================

Major improvements:
1. Capacity-aware Entity Scaling (CES) post-training
2. Grid search KGE
3. Evaluation pre and post CES

Autor: OBOE team!
Fecha: Nov 2025
"""

import pandas as pd
import numpy as np
from pykeen.pipeline import pipeline as pykeen_pipeline
from pykeen.triples import TriplesFactory
from pykeen.evaluation import RankBasedEvaluator
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    log_loss, roc_auc_score
)
from xgboost import XGBClassifier
from scipy import stats
import torch
import joblib
import os
import json
from datetime import datetime
from typing import Tuple, Dict, Any, Optional, List
from collections import Counter
import warnings
import sys

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURACIONES GLOBALES
# ==============================================================================

REPOSITORIES = {
    'amazon': 'dataset_triplet_amazon_new_simplificado.csv',
    'bbc': 'dataset_triplet_bbc_new_simplificado.csv',
    'reuters_activities': 'dataset_triplet_reuters_activities_new_simplificado.csv'
}

MODELS_AVAILABLE = ['transe', 'convkb', 'complex', 'distmult']

# Alpha values para CES (según Guan et al.)
CES_ALPHA_VALUES = [0.3, 0.5, 0.7, 1.0]  # De suave a fuerte


# ==============================================================================
# CLASE: CES STRATEGY (Guan et al., 2025)
# ==============================================================================

class CapacityAwareScaling:
    """
    Implementación de la estrategia CES de Guan et al. (2025)

    "Low-frequency entities are overfitted by large embedding spaces;
    regularizing or shrinking their embedding size improves generalization."
    """

    def __init__(self, model, triples, alpha: float = 0.5):
        """
        Args:
            model: Modelo PyKEEN entrenado
            triples: numpy array de triples [head, relation, tail]
            alpha: Exponente de reescalado (0.3=suave, 1.0=fuerte)
        """
        self.model = model
        self.triples = triples
        self.alpha = alpha
        self.entity_counts = None
        self.max_freq = None
        self.original_embeddings = None
        self.scaled_embeddings = None

    def count_entity_frequencies(self):
        """Contar frecuencia de entidades (head + tail)"""
        print(f"\n[CES] Counting entity frequencies (alpha={self.alpha})...")

        # Contar heads y tails
        self.entity_counts = Counter(self.triples[:, 0]) + Counter(self.triples[:, 2])
        self.max_freq = max(self.entity_counts.values())

        # Estadísticas
        freq_values = list(self.entity_counts.values())
        print(f"  Total entities: {len(self.entity_counts)}")
        print(f"  Max frequency: {self.max_freq}")
        print(f"  Mean frequency: {np.mean(freq_values):.2f}")
        print(f"  Median frequency: {np.median(freq_values):.2f}")
        print(f"  Min frequency: {min(freq_values)}")

        # Identificar entidades raras (< 10% de max_freq)
        rare_threshold = self.max_freq * 0.1
        rare_entities = sum(1 for f in freq_values if f < rare_threshold)
        print(
            f"  Rare entities (<{rare_threshold:.0f} occurrences): {rare_entities} ({100 * rare_entities / len(freq_values):.1f}%)")

        return self.entity_counts

    def scale_embeddings(self):
        """
        Aplicar reescalado CES:
        scale = (freq / max_freq) ** alpha

        - Alpha bajo (0.3-0.5): reescalado suave
        - Alpha alto (0.7-1.0): reescalado fuerte
        """
        print(f"\n[CES] Applying capacity-aware scaling (alpha={self.alpha})...")

        # Obtener embeddings originales
        entity_emb = self.model.entity_representations[0]()
        self.original_embeddings = entity_emb.detach().cpu().numpy()

        num_entities = len(self.original_embeddings)
        print(f"  Embedding shape: {self.original_embeddings.shape}")

        # Aplicar reescalado
        scaled_embs = []
        scale_factors = []

        for entity_id in range(num_entities):
            freq = self.entity_counts.get(entity_id, 1)
            scale = (freq / self.max_freq) ** self.alpha
            scale_factors.append(scale)

            scaled_emb = self.original_embeddings[entity_id] * scale
            scaled_embs.append(scaled_emb)

        self.scaled_embeddings = np.array(scaled_embs)

        # Estadísticas de reescalado
        scale_factors = np.array(scale_factors)
        print(f"  Scale factors - Mean: {scale_factors.mean():.3f}, Std: {scale_factors.std():.3f}")
        print(f"  Scale factors - Min: {scale_factors.min():.3f}, Max: {scale_factors.max():.3f}")

        # Calcular cambio en normas
        orig_norms = np.linalg.norm(self.original_embeddings, axis=1)
        scaled_norms = np.linalg.norm(self.scaled_embeddings, axis=1)
        norm_reduction = ((orig_norms - scaled_norms) / orig_norms * 100)
        print(f"  Norm reduction - Mean: {norm_reduction.mean():.1f}%, Max: {norm_reduction.max():.1f}%")

        return self.scaled_embeddings

    def apply_to_model(self):
        """Reemplazar embeddings en el modelo"""
        print(f"\n[CES] Replacing embeddings in model...")

        with torch.no_grad():
            self.model.entity_representations[0]._embeddings.weight.copy_(
                torch.tensor(self.scaled_embeddings, dtype=torch.float32)
            )

        print(f"  ✓ Entity embeddings rescaled successfully")

    def apply_ces(self):
        """Pipeline completo de CES"""
        self.count_entity_frequencies()
        self.scale_embeddings()
        self.apply_to_model()

        return self.model


# ==============================================================================
# CLASE: KGE OPTIMIZER CON CES
# ==============================================================================

class KGEOptimizerWithCES:
    """
    Optimizador de KGE con:
    - Selección por downstream AUC
    - CES post-training (Guan et al.)
    - Grid search ajustado según resultados MRR
    """

    AVAILABLE_MODELS = {
        'transe': 'TransE',
        'convkb': 'ConvKB',
        'complex': 'ComplEx',
        'distmult': 'DistMult'
    }

    def __init__(self, kg_path: str, emb_dir: str, output_dir: str,
                 model_type: str = 'complex',
                 search_type: str = 'smart',
                 apply_ces: bool = True,
                 ces_alpha: float = 0.5):

        self.kg_csv_path = kg_path
        self.emb_dir = emb_dir
        self.output_dir = output_dir
        self.model_type = model_type.lower()
        self.search_type = search_type
        self.apply_ces = apply_ces
        self.ces_alpha = ces_alpha

        self.model = None
        self.entity_to_id = None
        self.complete_entity_to_id = None
        self.relation_to_id = None
        self.best_embedding_dim = None
        self.training_triples = None  # Para CES

        self.all_trained_models = []

        os.makedirs(emb_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_file = os.path.join(output_dir, f"{model_type}_results_{timestamp}.json")
        self.metrics_file = os.path.join(output_dir, f"{model_type}_metrics_{timestamp}.txt")

    def prepare_triples_factory(self) -> Tuple[TriplesFactory, TriplesFactory]:
        """Preparar train y test splits"""
        print("\nPreparing triples factory...")
        df = pd.read_csv(self.kg_csv_path)
        triples = df[['subject', 'relation', 'object']].values

        create_inverse = (self.model_type == 'complex')
        tf = TriplesFactory.from_labeled_triples(
            triples,
            create_inverse_triples=create_inverse
        )

        training_tf, testing_tf = tf.split([0.8, 0.2], random_state=42)

        # Guardar para CES
        self.training_triples = training_tf.mapped_triples.cpu().numpy()

        # Guardar entity_to_id completo (todos los embeddings)
        self.complete_entity_to_id = tf.entity_to_id

        print(f"Total triples: {len(triples)}")
        print(f"Training triples: {len(training_tf.triples)} (80%)")
        print(f"Testing triples: {len(testing_tf.triples)} (20%)")
        print(f"Entities: {training_tf.num_entities}")
        print(f"Relations: {training_tf.num_relations}")

        return training_tf, testing_tf

    # ========================================================================
    # SMART SEARCH AJUSTADO (basado en resultados MRR)
    # ========================================================================

    def _get_smart_search_transe(self) -> List[Dict]:
        """
        TransE: MRR test razonable (0.16-0.30), gap moderado (0.39-0.45)
        Estrategia: Dimensiones bajas-medias, regularización moderada
        """
        combinations = []

        # Basado en tus resultados: amazon (gap=0.45), bbc (gap=0.39)
        embedding_dims = [16, 24, 32]  # Expandir un poco

        for dim in embedding_dims:
            for reg in [0.05]:  # Variar regularización
                config = {
                    'embedding_dim': dim,
                    'learning_rate': 0.001,
                    'num_epochs': 180,
                    'batch_size': 128,
                    'scoring_fct_norm': 1,
                    'regularizer_weight': reg,
                }
                combinations.append(config)

        print("\nSMART SEARCH - TransE (ajustado)")
        print(f"  Configs: {len(combinations)}")
        print(f"  Embedding dims: {embedding_dims}")
        print(f"  Regularization: [0.03, 0.05, 0.07]")

        return combinations

    def _get_smart_search_convkb(self) -> List[Dict]:
        """
        ConvKB: MRR test bajo (0.08-0.01), gap bajo pero pobre generalización
        Estrategia: Dimensiones MUY bajas, alta regularización, más dropout
        """
        combinations = []

        embedding_dims = [6, 8, 12]  # Muy bajo

        for dim in embedding_dims:
            for filters in [4, 6]:
                config = {
                    'embedding_dim': dim,
                    'learning_rate': 0.002,
                    'num_epochs': 150,
                    'batch_size': 128,
                    'num_filters': 6,
                    'hidden_dropout_rate': 0.5,
                    'regularizer_weight': 0.01,  # Fuerte
                }
                combinations.append(config)

        print("\nSMART SEARCH - ConvKB (ajustado)")
        print(f"  Configs: {len(combinations)}")
        print(f"  Embedding dims: {embedding_dims} (muy bajo)")
        print(f"  High dropout & regularization")

        return combinations

    def _get_smart_search_complex(self) -> List[Dict]:
        """
        ComplEx: MRR test bajo (0.10-0.12), gap ALTO (0.44-0.51) - OVERFITTING SEVERO
        Estrategia: Dimensiones bajas, regularización FUERTE, menos epochs
        """
        combinations = []

        # Tus resultados muestran overfitting severo
        # Reducir capacidad drásticamente
        configs_raw = [
            # (dim, lr, epochs, batch, reg)
            # (12, 0.01, 150, 128, 0.04),  # Dim muy bajo, reg alta
            # (12, 0.01, 150, 128, 0.05),  # Dim muy bajo, reg alta
            (16, 0.01, 180, 128, 0.025),
            (24, 0.015, 150, 128, 0.025),
            (24, 0.02, 180, 128, 0.45),
            (32, 0.015, 150, 128, 0.045),

        ]

        for dim, lr, epochs, batch, reg in configs_raw:
            config = {
                'embedding_dim': dim,
                'learning_rate': lr,
                'num_epochs': epochs,
                'batch_size': batch,
                'regularizer_weight': reg,
            }
            combinations.append(config)

        print("\nSMART SEARCH - ComplEx (anti-overfitting)")
        print(f"  Configs: {len(combinations)}")
        print(f"  Dims: [12-32] (reducido vs original 128)")
        print(f"  Strong regularization: 0.02-0.05")
        print(f"  Fewer epochs: 150-180")

        return combinations

    def _get_smart_search_distmult(self) -> List[Dict]:
        """
        DistMult: MRR test bajo (0.02-0.08), gap moderado (0.09-0.10)
        Estrategia: Similar a ConvKB, dimensiones bajas
        """
        combinations = []

        embedding_dims = [8, 12, 16, 20]

        for dim in embedding_dims:
            for reg in [0.05, 0.075, 0.1]:
                config = {
                    'embedding_dim': dim,
                    'learning_rate': 0.001,
                    'num_epochs': 250,
                    'batch_size': 128,
                    'regularizer_weight': reg,
                }
                combinations.append(config)

        print("\nSMART SEARCH - DistMult (ajustado)")
        print(f"  Configs: {len(combinations)}")
        print(f"  Embedding dims: {embedding_dims}")
        print(f"  Strong regularization: [0.05, 0.075, 0.1]")

        return combinations

    def _get_grid_search_configs(self) -> List[Dict]:
        """Grid search ajustado según modelo"""
        if self.model_type == 'complex':
            # Reducir grid para ComplEx (mostró mayor overfitting)
            embedding_dims = [12, 24, 32]
            learning_rates = [0.01, 0.02]
            num_epochs_list = [150, 200]
            batch_sizes = [128, 256]
            regularizer_weights = [0.02, 0.04, 0.06]

        elif self.model_type == 'transe':
            embedding_dims = [16, 24, 32, 48]
            learning_rates = [0.0005, 0.001]
            num_epochs_list = [120, 150]
            batch_sizes = [128]
            regularizer_weights = [0.03, 0.05, 0.07]

        else:  # convkb, distmult
            embedding_dims = [8, 12, 16, 24]
            learning_rates = [0.001, 0.002]
            num_epochs_list = [150, 200]
            batch_sizes = [128]
            regularizer_weights = [0.05, 0.075, 0.1]

        combinations = []
        for dim in embedding_dims:
            for lr in learning_rates:
                for epochs in num_epochs_list:
                    for batch in batch_sizes:
                        for reg in regularizer_weights:
                            config = {
                                'embedding_dim': dim,
                                'learning_rate': lr,
                                'num_epochs': epochs,
                                'batch_size': batch,
                                'regularizer_weight': reg,
                            }
                            combinations.append(config)

        print(f"\nGRID SEARCH - {self.model_type.upper()} (ajustado)")
        print(f"  Total configs: {len(combinations)}")
        print(f"  Embedding dims: {embedding_dims}")

        return combinations

    def get_search_configurations(self) -> List[Dict]:
        """Obtener configuraciones según search_type"""
        if self.search_type == 'smart':
            if self.model_type == 'transe':
                return self._get_smart_search_transe()
            elif self.model_type == 'convkb':
                return self._get_smart_search_convkb()
            elif self.model_type == 'complex':
                return self._get_smart_search_complex()
            elif self.model_type == 'distmult':
                return self._get_smart_search_distmult()
        elif self.search_type == 'grid':
            return self._get_grid_search_configs()
        else:
            raise ValueError(f"Unknown search type: {self.search_type}")

    # ========================================================================
    # TRAIN CON CES
    # ========================================================================

    def train_single_config(self, training_tf: TriplesFactory, testing_tf: TriplesFactory,
                            config: Dict) -> Tuple[Any, Dict]:
        """Entrenar una configuración y aplicar CES si está activado"""
        print(f"\nTraining config: {config}")

        embedding_dim = config['embedding_dim']
        learning_rate = config['learning_rate']
        num_epochs = config['num_epochs']
        batch_size = config['batch_size']
        regularizer_weight = config.get('regularizer_weight', 1e-5)

        model_kwargs = {'embedding_dim': embedding_dim}

        if self.model_type == 'transe':
            model_kwargs.update({
                'scoring_fct_norm': config.get('scoring_fct_norm', 1),
            })
        elif self.model_type == 'convkb':
            model_kwargs.update({
                'num_filters': config.get('num_filters', 32),
                'hidden_dropout_rate': config.get('hidden_dropout_rate', 0.5),
            })

        negative_sampler = 'basic' if self.model_type != 'transe' else 'bernoulli'
        negative_sampler_kwargs = dict(num_negs_per_pos=1)

        device = 'mps' if self.model_type in ['convkb', 'transe', 'distmult'] else 'cpu'
        print(f"Using device: {device}")

        training_sub_tf, validation_tf = training_tf.split([0.9, 0.1], random_state=42)

        stopper_kwargs = {
            'frequency': 5,
            'patience': 5,
            'relative_delta': 0.0001,
            'metric': 'mean_reciprocal_rank',
        }

        # Train
        try:
            result = pykeen_pipeline(
                training=training_sub_tf,
                validation=validation_tf,
                testing=testing_tf,
                model=self.model_type,
                device=device,
                model_kwargs=model_kwargs,
                negative_sampler=negative_sampler,
                negative_sampler_kwargs=negative_sampler_kwargs,
                optimizer='adam',
                optimizer_kwargs=dict(lr=learning_rate),
                regularizer='lp' if self.model_type != 'convkb' else None,
                regularizer_kwargs=dict(weight=regularizer_weight, p=2) if self.model_type != 'convkb' else None,
                loss='nssa' if self.model_type == 'complex' else 'negativeloglikelihood',
                loss_kwargs=dict(adversarial_temperature=1.0) if self.model_type == 'complex' else {},
                training_kwargs=dict(
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                    use_tqdm_batch=False,
                    use_tqdm=True,
                ),
                stopper='early',
                stopper_kwargs=stopper_kwargs,
                evaluator_kwargs=dict(filtered=True),
                evaluation_kwargs=dict(batch_size=256),
                random_seed=42,
            )
        except Exception as e:
            print(f"ERROR training: {e}")
            return None, {'error': str(e), 'config': config}

        # ===== EVALUACIÓN PRE-CES =====
        evaluator = RankBasedEvaluator(filtered=True)

        test_metrics_pre = {
            'mrr': float(result.metric_results.get_metric('mean_reciprocal_rank')),
            'hits_at_1': float(result.metric_results.get_metric('hits@1')),
            'hits_at_10': float(result.metric_results.get_metric('hits@10')),
        }

        train_results_pre = evaluator.evaluate(
            model=result.model,
            mapped_triples=training_tf.mapped_triples,
            additional_filter_triples=[training_tf.mapped_triples],
            batch_size=batch_size,
            use_tqdm=False
        )
        train_metrics_pre = {
            'mrr': float(train_results_pre.get_metric('mean_reciprocal_rank')),
            'hits_at_1': float(train_results_pre.get_metric('hits@1')),
            'hits_at_10': float(train_results_pre.get_metric('hits@10')),
        }

        gap_pre = train_metrics_pre['mrr'] - test_metrics_pre['mrr']

        print(f"\n  PRE-CES:")
        print(f"    Train MRR: {train_metrics_pre['mrr']:.4f}")
        print(f"    Test MRR:  {test_metrics_pre['mrr']:.4f}")
        print(f"    Gap: {gap_pre:.4f}")
        self.save_embeddings_universal(result.model, self.emb_dir, self.complete_entity_to_id, suffix="preces")

        # ===== APLICAR CES =====
        ces_applied = False
        train_metrics_post = train_metrics_pre
        test_metrics_post = test_metrics_pre
        gap_post = gap_pre
        gap_improvement = 0.0

        if self.apply_ces:
            try:
                print(f"\n  Applying CES (alpha={self.ces_alpha})...")
                ces = CapacityAwareScaling(
                    model=result.model,
                    triples=self.training_triples,
                    alpha=self.ces_alpha
                )
                result.model = ces.apply_ces()
                ces_applied = True

                # Re-evaluar POST-CES
                test_results_post = evaluator.evaluate(
                    model=result.model,
                    mapped_triples=testing_tf.mapped_triples,
                    additional_filter_triples=[training_tf.mapped_triples, testing_tf.mapped_triples],
                    batch_size=batch_size,
                    use_tqdm=False
                )
                test_metrics_post = {
                    'mrr': float(test_results_post.get_metric('mean_reciprocal_rank')),
                    'hits_at_1': float(test_results_post.get_metric('hits@1')),
                    'hits_at_10': float(test_results_post.get_metric('hits@10')),
                }

                train_results_post = evaluator.evaluate(
                    model=result.model,
                    mapped_triples=training_tf.mapped_triples,
                    additional_filter_triples=[training_tf.mapped_triples],
                    batch_size=batch_size,
                    use_tqdm=False
                )
                train_metrics_post = {
                    'mrr': float(train_results_post.get_metric('mean_reciprocal_rank')),
                    'hits_at_1': float(train_results_post.get_metric('hits@1')),
                    'hits_at_10': float(train_results_post.get_metric('hits@10')),
                }

                gap_post = train_metrics_post['mrr'] - test_metrics_post['mrr']
                gap_improvement = gap_pre - gap_post
                self.save_embeddings_universal(result.model, self.emb_dir, self.complete_entity_to_id, suffix="postces")
                print(f"\n  POST-CES:")
                print(
                    f"    Train MRR: {train_metrics_post['mrr']:.4f} (Δ {train_metrics_post['mrr'] - train_metrics_pre['mrr']:+.4f})")
                print(
                    f"    Test MRR:  {test_metrics_post['mrr']:.4f} (Δ {test_metrics_post['mrr'] - test_metrics_pre['mrr']:+.4f})")
                print(
                    f"    Gap: {gap_post:.4f} (improvement: {gap_improvement:.4f}) {'✓' if gap_improvement > 0 else ''}")

            except Exception as e:
                print(f"  WARNING: CES failed: {e}")
                ces_applied = False

        # Compile metrics
        metrics = {
            **config,
            'model_type': self.model_type,
            'ces_applied': ces_applied,
            'ces_alpha': self.ces_alpha if ces_applied else None,

            # Pre-CES
            'pre_ces': {
                'train': train_metrics_pre,
                'test': test_metrics_pre,
                'gap': gap_pre
            },

            # Post-CES (si aplica)
            'post_ces': {
                'train': train_metrics_post,
                'test': test_metrics_post,
                'gap': gap_post
            } if ces_applied else None,

            'gap_improvement': gap_improvement if ces_applied else None,

            # Para selección (usar post-CES si está disponible)
            'test_mrr': test_metrics_post['mrr'],
            'train_test_gap': gap_post,
            'config_index': len(self.all_trained_models)
        }

        self.all_trained_models.append({
            'model': result.model,
            'metrics': metrics,
            'config_index': len(self.all_trained_models)
        })

        return result.model, metrics

    def save_embeddings_universal(self, model, emb_dir: str, entity_to_id: Dict[str, int], suffix: str = "preces"):
        """
        Guardar TODOS los embeddings de entidades para cualquier modelo PyKEEN.
        Compatible con TransE, DistMult, ComplEx y ConvKB.
        ============================================================

        NOTA: Esta función guarda todos los embeddings del modelo, no solo los de train.
        El modelo aprende embeddings para todas las entidades del dataset completo.

        suffix = 'preces' o 'postces'
        """
        os.makedirs(emb_dir, exist_ok=True)
        print(f"\n[SAVE] Exporting ALL {suffix.upper()} embeddings to {emb_dir} ...")

        embeddings = None

        # 1️⃣ Extraer embeddings de entidad
        try:
            entity_repr = model.entity_representations[0]()
            embeddings = entity_repr.detach().cpu().numpy()
        except Exception as e:
            print(f"  ⚠️ Could not extract entity embeddings directly: {e}")
            embeddings = None

        # 2️⃣ Si el modelo es ComplEx, concatenar real+imag
        model_name = model.__class__.__name__.lower()
        if embeddings is not None and "complex" in model_name:
            if np.iscomplexobj(embeddings):
                embeddings = np.concatenate([np.real(embeddings), np.imag(embeddings)], axis=1)
                print("  ✓ Complex embeddings flattened (real + imag)")

        # 3️⃣ Si el modelo es ConvKB, intentar acceder a embedding interno
        elif embeddings is None and "convkb" in model_name:
            try:
                embeddings = model.entity_embeddings.weight.detach().cpu().numpy()
                print("  ✓ ConvKB entity embeddings extracted")
            except Exception as e:
                print(f"  ⚠️ ConvKB embeddings unavailable: {e}")
                embeddings = None

        if embeddings is None:
            print("  ⚠️ No embeddings extracted. Skipping save.")
            return

        # 4️⃣ Guardar numpy
        np.save(os.path.join(emb_dir, f"embeddings_{suffix}.npy"), embeddings)

        # 5️⃣ Guardar CSV legible si pequeño
        if embeddings.shape[0] <= 5000:
            pd.DataFrame(embeddings).to_csv(os.path.join(emb_dir, f"embeddings_{suffix}.csv"), index=False)

        # 6️⃣ Guardar mapping y stats
        with open(os.path.join(emb_dir, "entity_mapping.json"), "w") as f:
            json.dump(entity_to_id, f, indent=2)

        stats = {
            "num_entities": len(embeddings),
            "embedding_dim": embeddings.shape[1],
            "model_type": model.__class__.__name__,
            "suffix": suffix,
        }
        with open(os.path.join(emb_dir, f"embedding_stats_{suffix}.json"), "w") as f:
            json.dump(stats, f, indent=2)

        print(f"  ✓ Embeddings ({suffix}) saved successfully ({embeddings.shape})")

    def train_all_configs(self, training_tf: TriplesFactory, testing_tf: TriplesFactory) -> Dict:
        """Entrenar todas las configuraciones"""
        print("\n" + "=" * 80)
        print(f"TRAINING ALL CONFIGURATIONS - {self.model_type.upper()} ({self.search_type.upper()})")
        if self.apply_ces:
            print(f"CES ENABLED (alpha={self.ces_alpha})")
        print("=" * 80)

        configs = self.get_search_configurations()
        print(f"\nTotal configurations to train: {len(configs)}")

        for i, config in enumerate(configs, 1):
            print(f"\n[{i}/{len(configs)}] Training configuration...")
            model, metrics = self.train_single_config(training_tf, testing_tf, config)

        print("\n" + "=" * 80)
        print("ALL CONFIGURATIONS TRAINED")
        successful = [m for m in self.all_trained_models if 'error' not in m['metrics']]
        print(f"Total successful: {len(successful)}")

        if self.apply_ces and successful:
            # Analizar efectividad CES
            improvements = [m['metrics']['gap_improvement'] for m in successful
                            if m['metrics'].get('gap_improvement') is not None]
            if improvements:
                print(f"\nCES Impact Analysis:")
                print(f"  Configs with gap improvement: {sum(1 for x in improvements if x > 0)}/{len(improvements)}")
                print(f"  Mean gap improvement: {np.mean(improvements):.4f}")
                print(f"  Max gap improvement: {max(improvements):.4f}")

        print("=" * 80)

        return {'all_models': self.all_trained_models}

    def save_results(self, metrics: Dict):
        """Guardar resultados"""
        with open(self.results_file, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)

        with open(self.metrics_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"KGE OPTIMIZATION RESULTS - {self.model_type.upper()}\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Search type: {self.search_type}\n")
            f.write(f"CES applied: {self.apply_ces}\n")
            if self.apply_ces:
                f.write(f"CES alpha: {self.ces_alpha}\n")
            f.write(f"Total configurations trained: {len(self.all_trained_models)}\n")

        print(f"Results saved to: {self.results_file}")


# ==============================================================================
# CLASE 2: CLASSIFICATION PIPELINE CON CES
# ==============================================================================

class IntegratedClassificationPipeline:
    """
    Pipeline completo: KGE + Clasificación con CES
    - Selección por downstream AUC
    - Usa embeddings POST-CES para mejor generalización
    """

    def __init__(self, kg_path: str, emb_dir: str, output_dir: str,
                 model_type: str = 'complex',
                 search_type: str = 'smart',
                 n_folds: int = 10,
                 apply_ces: bool = True,
                 ces_alpha: float = 0.5):

        self.kge_csv_path = kg_path
        self.emb_dir = emb_dir
        self.output_dir = output_dir
        self.model_type = model_type
        self.search_type = search_type
        self.n_folds = n_folds

        # KGE optimizer CON CES
        self.kge_optimizer = KGEOptimizerWithCES(
            kg_path, emb_dir, output_dir,
            model_type, search_type,
            apply_ces=apply_ces,
            ces_alpha=ces_alpha
        )

        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_file = os.path.join(output_dir, f"classification_results_{timestamp}.json")
        self.statistical_file = os.path.join(output_dir, f"statistical_analysis_{timestamp}.txt")
        self.model_file = os.path.join(output_dir, f"{model_type}_best_classifier.pkl")

        df = pd.read_csv(kg_path)
        self.is_binary_classification = df['topic'].nunique() == 2
        self.embedding_dim = None

    # ========================================================================
    # EXTRACT EMBEDDINGS (con fix ComplEx y nota sobre CES)
    # ========================================================================

    def extract_embeddings(self, model: Any, entity_to_id: Dict, df: pd.DataFrame) -> np.ndarray:
        """
        Extraer embeddings con FIX para ComplEx (números complejos)

        IMPORTANTE: Si CES fue aplicado, este método extrae los embeddings
        POST-CES del modelo, que son los que deben usarse downstream según
        Guan et al. (2025). Los embeddings reescalados generalizan mejor.
        """
        print("\nExtracting embeddings from KGE model...")
        if hasattr(self, 'kge_optimizer') and self.kge_optimizer.apply_ces:
            print("  ✓ Using POST-CES embeddings (reescaled for better generalization)")

        X_list = []
        for idx, row in df.iterrows():
            subject_emb = self._get_entity_embedding(model, entity_to_id, row['subject'])
            object_emb = self._get_entity_embedding(model, entity_to_id, row['object'])

            combined_emb = np.concatenate([subject_emb, object_emb])
            X_list.append(combined_emb)

            if (idx + 1) % 1000 == 0:
                print(f"  Processed {idx + 1}/{len(df)} samples...")

        X = np.array(X_list)
        print(f"Feature matrix shape: {X.shape}")

        return X

    def save_complete_embeddings_for_visualization(self, X: np.ndarray, df: pd.DataFrame,
                                                   entity_to_id: Dict, suffix: str = "complete"):
        """
        Guardar embeddings COMPLETOS del dataset para visualización UMAP.
        Estos embeddings ya están procesados (subject + object concatenados)
        y corresponden exactamente a las filas del DataFrame.

        Args:
            X: Matriz de embeddings (n_samples, 2*embedding_dim)
            df: DataFrame original con todos los datos
            entity_to_id: Mapeo de entidades a IDs
            suffix: Sufijo para los archivos
        """
        emb_dir = self.emb_dir
        os.makedirs(emb_dir, exist_ok=True)

        print(f"\n[SAVE COMPLETE] Exporting embeddings for UMAP visualization...")
        print(f"  Shape: {X.shape} (matches DataFrame rows: {len(df)})")

        # 1. Guardar embeddings concatenados (listos para UMAP)
        np.save(os.path.join(emb_dir, f"embeddings_{suffix}_umap.npy"), X)

        # 2. Guardar topics correspondientes
        topics = df['topic'].values
        np.save(os.path.join(emb_dir, f"topics_{suffix}.npy"), topics)

        # 3. Guardar información de subjects y objects
        metadata = {
            'subjects': df['subject'].tolist(),
            'objects': df['object'].tolist(),
            'relations': df['relation'].tolist(),
            'topics': df['topic'].tolist()
        }
        with open(os.path.join(emb_dir, f"metadata_{suffix}.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        # 4. Guardar entity mapping completo
        with open(os.path.join(emb_dir, "entity_mapping_complete.json"), "w") as f:
            json.dump(entity_to_id, f, indent=2)

        # 5. Guardar estadísticas
        unique_topics, counts = np.unique(topics, return_counts=True)
        stats = {
            "num_samples": int(X.shape[0]),
            "embedding_dim": int(X.shape[1]),
            "num_unique_topics": int(len(unique_topics)),
            "topics_distribution": {str(topic): int(count) for topic, count in zip(unique_topics, counts)},
            "num_entities": len(entity_to_id),
            "dataset_shape": [int(x) for x in df.shape],
            "suffix": suffix,
            "description": "Complete embeddings for UMAP visualization (subject+object concatenated)"
        }
        with open(os.path.join(emb_dir, f"embedding_stats_{suffix}.json"), "w") as f:
            json.dump(stats, f, indent=2)

        print(f"  ✅ Complete embeddings saved successfully")
        print(f"  ✅ Topics saved: {len(topics)} labels")
        print(f"  ✅ Perfect match for UMAP: X.shape[0] == len(topics) == {len(df)}")

    def _get_entity_embedding(self, model: Any, entity_to_id: Dict, entity: str) -> np.ndarray:
        """
        Obtener embedding con FIX para ComplEx
        NOTA: Si CES fue aplicado, estos son los embeddings reescalados
        """
        if self.embedding_dim is None:
            sample_id = list(entity_to_id.values())[0]
            sample_tensor = torch.tensor([sample_id], dtype=torch.long)
            with torch.no_grad():
                sample_emb = model.entity_representations[0](sample_tensor).cpu().numpy().flatten()
            if np.iscomplexobj(sample_emb):
                self.embedding_dim = len(sample_emb)
            else:
                self.embedding_dim = len(sample_emb)

        if entity not in entity_to_id:
            if self.model_type == 'complex':
                return np.zeros(self.embedding_dim * 2)
            else:
                return np.zeros(self.embedding_dim)

        entity_id = entity_to_id[entity]
        entity_tensor = torch.tensor([entity_id], dtype=torch.long)

        with torch.no_grad():
            embedding = model.entity_representations[0](entity_tensor)

        embedding_np = embedding.cpu().numpy().flatten()

        # FIX PARA COMPLEX
        if np.iscomplexobj(embedding_np):
            embedding_real = np.real(embedding_np)
            embedding_imag = np.imag(embedding_np)
            embedding_np = np.concatenate([embedding_real, embedding_imag])

        return embedding_np

    # ========================================================================
    # SELECCIÓN POR DOWNSTREAM AUC (con embeddings POST-CES)
    # ========================================================================

    def select_best_model_by_auc(self, df: pd.DataFrame, y: np.ndarray) -> Tuple[Any, Dict, np.ndarray]:
        """
        SELECCIÓN POR DOWNSTREAM AUC (NO por MRR)
        IMPORTANTE: Usa embeddings POST-CES si CES fue aplicado
        """
        print("\n" + "=" * 80)
        print("MODEL SELECTION: Evaluating by DOWNSTREAM AUC (NOT MRR)")
        if self.kge_optimizer.apply_ces:
            print("NOTE: Using POST-CES embeddings for evaluation")
        print("=" * 80)

        model_performances = []

        for i, model_info in enumerate(self.kge_optimizer.all_trained_models, 1):
            if 'error' in model_info['metrics']:
                print(f"\n[{i}] Skipping failed model")
                continue

            metrics = model_info['metrics']
            print(f"\n[{i}/{len(self.kge_optimizer.all_trained_models)}] Evaluating model...")
            print(f"  KGE Test MRR: {metrics['test_mrr']:.4f}")
            print(f"  KGE Train-Test Gap: {metrics['train_test_gap']:.4f}")

            if metrics.get('ces_applied'):
                print(f"  CES Gap Improvement: {metrics['gap_improvement']:.4f} ✓")

            # Extract embeddings (POST-CES si fue aplicado)
            X_temp = self.extract_embeddings(
                model_info['model'],
                self.kge_optimizer.entity_to_id,
                df
            )

            X_train, X_test, y_train, y_test = train_test_split(
                X_temp, y, test_size=0.2, random_state=42, stratify=y
            )

            clf = XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                objective='binary:logistic' if self.is_binary_classification else 'multi:softprob',
                random_state=42,
                use_label_encoder=False
            )

            clf.fit(X_train, y_train)
            y_pred_proba = clf.predict_proba(X_test)

            if self.is_binary_classification:
                auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
            else:
                auc_score = roc_auc_score(y_test, y_pred_proba,
                                          multi_class='ovr', average='macro')

            print(f"  Downstream AUC: {auc_score:.4f} ← SELECTION METRIC")

            model_performances.append({
                'model': model_info['model'],
                'metrics': metrics,
                'downstream_auc': auc_score,
                'config_index': model_info['config_index']
            })

        if not model_performances:
            raise ValueError("No valid models trained!")

        best_by_auc = max(model_performances, key=lambda x: x['downstream_auc'])

        print("\n" + "=" * 80)
        print("BEST MODEL SELECTED BY DOWNSTREAM AUC")
        print("=" * 80)
        print(f"Configuration index: {best_by_auc['config_index']}")
        print(f"KGE Test MRR: {best_by_auc['metrics']['test_mrr']:.4f}")
        print(f"KGE Train-Test Gap: {best_by_auc['metrics']['train_test_gap']:.4f}")
        if best_by_auc['metrics'].get('ces_applied'):
            print(f"CES Gap Improvement: {best_by_auc['metrics']['gap_improvement']:.4f}")
        print(f"Downstream AUC: {best_by_auc['downstream_auc']:.4f} ← SELECTION CRITERION")

        self.kge_optimizer.model = best_by_auc['model']
        self.kge_optimizer.best_embedding_dim = best_by_auc['metrics']['embedding_dim']
        self.embedding_dim = best_by_auc['metrics']['embedding_dim']

        # Re-extract con mejor modelo (POST-CES)
        X_best = self.extract_embeddings(
            best_by_auc['model'],
            self.kge_optimizer.entity_to_id,
            df
        )

        # GUARDAR EMBEDDINGS COMPLETOS PARA UMAP
        self.save_complete_embeddings_for_visualization(
            X_best,
            df,
            self.kge_optimizer.entity_to_id,
            suffix="complete_postces" if best_by_auc['metrics'].get('ces_applied') else "complete_preces"
        )
        print(f"\n✅ Complete embeddings saved for UMAP visualization")
        print(f"   Use 'embeddings_complete_*.npy' and 'topics_complete.npy' for UMAP")

        return best_by_auc['model'], best_by_auc['metrics'], X_best

    # ========================================================================
    # CROSS-VALIDATION
    # ========================================================================

    def perform_cross_validation(self, X: np.ndarray, y: np.ndarray,
                                 best_params: Dict) -> Dict:
        """10-fold CV con todas las métricas"""
        print(f"\nPerforming {self.n_folds}-fold cross-validation...")

        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        fold_accuracies = []
        fold_aucs = []
        fold_logloss = []
        fold_mean_per_class_error = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            print(f"  Fold {fold}/{self.n_folds}...", end=' ')

            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            model = XGBClassifier(
                **best_params,
                objective='binary:logistic' if self.is_binary_classification else 'multi:softprob',
                eval_metric='logloss',
                random_state=42,
                use_label_encoder=False
            )

            model.fit(X_train_fold, y_train_fold)

            y_pred = model.predict(X_val_fold)
            y_pred_proba = model.predict_proba(X_val_fold)

            accuracy = accuracy_score(y_val_fold, y_pred)
            fold_accuracies.append(accuracy)

            if self.is_binary_classification:
                auc_score = roc_auc_score(y_val_fold, y_pred_proba[:, 1])
            else:
                auc_score = roc_auc_score(y_val_fold, y_pred_proba,
                                          multi_class='ovr', average='macro')
            fold_aucs.append(auc_score)

            logloss_score = log_loss(y_val_fold, y_pred_proba)
            fold_logloss.append(logloss_score)

            cm = confusion_matrix(y_val_fold, y_pred)
            per_class_errors = []
            for i in range(len(cm)):
                if cm[i].sum() > 0:
                    recall = cm[i, i] / cm[i].sum()
                    error = 1 - recall
                    per_class_errors.append(error)
            mean_per_class_error = np.mean(per_class_errors) if per_class_errors else 0.0
            fold_mean_per_class_error.append(mean_per_class_error)

            print(f"Acc: {accuracy:.4f}, AUC: {auc_score:.4f}")

        mean_accuracy = np.mean(fold_accuracies)
        std_accuracy = np.std(fold_accuracies)
        mean_auc = np.mean(fold_aucs)
        std_auc = np.std(fold_aucs)
        mean_logloss = np.mean(fold_logloss)
        std_logloss = np.std(fold_logloss)
        mean_mpce = np.mean(fold_mean_per_class_error)
        std_mpce = np.std(fold_mean_per_class_error)

        ci_accuracy = stats.t.interval(0.95, len(fold_accuracies) - 1,
                                       loc=mean_accuracy, scale=stats.sem(fold_accuracies))
        ci_auc = stats.t.interval(0.95, len(fold_aucs) - 1,
                                  loc=mean_auc, scale=stats.sem(fold_aucs))

        print(f"\nCross-validation results:")
        print(
            f"  Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f} (95% CI: {ci_accuracy[0]:.4f}–{ci_accuracy[1]:.4f})")
        print(f"  AUC: {mean_auc:.4f} ± {std_auc:.4f} (95% CI: {ci_auc[0]:.4f}–{ci_auc[1]:.4f})")
        print(f"  LogLoss: {mean_logloss:.4f} ± {std_logloss:.4f}")
        print(f"  MPCE: {mean_mpce:.4f} ± {std_mpce:.4f}")

        return {
            'fold_accuracies': fold_accuracies,
            'fold_aucs': fold_aucs,
            'fold_logloss': fold_logloss,
            'fold_mean_per_class_error': fold_mean_per_class_error,
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'ci_accuracy': ci_accuracy,
            'mean_auc': mean_auc,
            'std_auc': std_auc,
            'ci_auc': ci_auc,
            'mean_logloss': mean_logloss,
            'std_logloss': std_logloss,
            'mean_mpce': mean_mpce,
            'std_mpce': std_mpce,
        }

    # ========================================================================
    # RUN CLASSIFICATION (con embeddings POST-CES)
    # ========================================================================

    def run_classification_with_statistics(self, n_iter: int = 20, cv: int = 3) -> Tuple[Any, Dict]:
        """Pipeline completo con CES y estadísticas"""
        print("\n" + "=" * 80)
        print("STARTING CLASSIFICATION WITH CES STRATEGY")
        print("=" * 80)

        df = pd.read_csv(self.kge_csv_path)
        y = df['topic'].values

        # Entrenar todas las configs KGE (con CES)
        training_tf, testing_tf = self.kge_optimizer.prepare_triples_factory()
        self.kge_optimizer.entity_to_id = self.kge_optimizer.complete_entity_to_id
        self.kge_optimizer.relation_to_id = training_tf.relation_to_id

        kge_results = self.kge_optimizer.train_all_configs(training_tf, testing_tf)

        # Seleccionar mejor modelo por downstream AUC (usa embeddings POST-CES)
        best_kge_model, best_kge_metrics, X = self.select_best_model_by_auc(df, y)

        print(f"\nFeature extraction completed")
        print(f"Feature shape: {X.shape}")
        if best_kge_metrics.get('ces_applied'):
            print(f"✓ Features extracted from POST-CES embeddings")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Hyperparameter search para XGBoost
        print("\nPerforming hyperparameter search for XGBoost...")

        param_distributions = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2]
        }

        base_model = XGBClassifier(
            objective='binary:logistic' if self.is_binary_classification else 'multi:softprob',
            eval_metric='logloss',
            random_state=42,
            use_label_encoder=False
        )

        search = RandomizedSearchCV(
            base_model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring='neg_log_loss',
            n_jobs=-1,
            verbose=1,
            random_state=42
        )

        search.fit(X_train, y_train)
        best_params = search.best_params_

        print(f"\nBest XGBoost parameters: {best_params}")

        # Cross-validation
        cv_results = self.perform_cross_validation(X, y, best_params)

        # Train final model
        print("\nTraining final model on full training set...")
        final_model = XGBClassifier(
            **best_params,
            objective='binary:logistic' if self.is_binary_classification else 'multi:softprob',
            eval_metric='logloss',
            random_state=42,
            use_label_encoder=False
        )
        final_model.fit(X_train, y_train)

        # Evaluate on test
        y_pred = final_model.predict(X_test)
        y_pred_proba = final_model.predict_proba(X_test)

        test_accuracy = accuracy_score(y_test, y_pred)
        if self.is_binary_classification:
            test_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        else:
            test_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')

        test_logloss = log_loss(y_test, y_pred_proba)

        test_cm = confusion_matrix(y_test, y_pred)
        test_per_class_errors = []
        for i in range(len(test_cm)):
            if test_cm[i].sum() > 0:
                recall = test_cm[i, i] / test_cm[i].sum()
                error = 1 - recall
                test_per_class_errors.append(error)
        test_mpce = np.mean(test_per_class_errors) if test_per_class_errors else 0.0

        print(f"\nTest set performance:")
        print(f"  Accuracy: {test_accuracy:.4f}")
        print(f"  AUC: {test_auc:.4f}")
        print(f"  LogLoss: {test_logloss:.4f}")
        print(f"  MPCE: {test_mpce:.4f}")

        # Wilcoxon test
        p_value = None
        if len(cv_results['fold_accuracies']) >= 2:
            mid = len(cv_results['fold_accuracies']) // 2
            group1 = cv_results['fold_accuracies'][:mid]
            group2 = cv_results['fold_accuracies'][mid:mid + len(group1)]

            if len(group1) == len(group2):
                statistic, p_value = stats.wilcoxon(group1, group2)
                print(f"\nWilcoxon test (p={p_value:.4f})")

        # Compile results
        final_metrics = {
            'kge_model_type': self.model_type,
            'kge_search_type': self.search_type,
            'kge_embedding_dim': best_kge_metrics['embedding_dim'],
            'ces_applied': best_kge_metrics.get('ces_applied', False),
            'ces_alpha': best_kge_metrics.get('ces_alpha'),
            'kge_metrics': {
                'pre_ces': best_kge_metrics.get('pre_ces'),
                'post_ces': best_kge_metrics.get('post_ces'),
                'gap_improvement': best_kge_metrics.get('gap_improvement'),
                'final_train_mrr': best_kge_metrics['post_ces']['train']['mrr'] if best_kge_metrics.get('post_ces') else
                best_kge_metrics['pre_ces']['train']['mrr'],
                'final_test_mrr': best_kge_metrics['test_mrr'],
                'final_train_test_gap': best_kge_metrics['train_test_gap'],
            },
            'selection_criterion': 'downstream_auc',
            'best_classifier_params': best_params,
            'cv_results': cv_results,
            'test_accuracy': test_accuracy,
            'test_auc': test_auc,
            'test_logloss': test_logloss,
            'test_mpce': test_mpce,
            'test_classification_report': classification_report(y_test, y_pred, output_dict=True),
            'test_confusion_matrix': test_cm.tolist(),
            'wilcoxon_p_value': p_value,
        }

        self.save_results(final_metrics, final_model)

        return final_model, final_metrics

    def save_results(self, metrics: Dict, model: Any):
        """Guardar resultados con info CES"""
        with open(self.results_file, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)

        with open(self.statistical_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("STATISTICAL ANALYSIS REPORT - WITH CES STRATEGY\n")
            f.write("=" * 80 + "\n\n")

            f.write("LEVEL 1: KGE METRICS (With CES Impact)\n")
            f.write("-" * 60 + "\n")
            f.write(f"Model: {metrics['kge_model_type'].upper()}\n")
            f.write(f"Search Type: {metrics['kge_search_type']}\n")
            f.write(f"Embedding Dim: {metrics['kge_embedding_dim']}\n")
            f.write(f"CES Applied: {metrics['ces_applied']}\n")
            if metrics['ces_applied']:
                f.write(f"CES Alpha: {metrics['ces_alpha']}\n\n")

                pre = metrics['kge_metrics']['pre_ces']
                post = metrics['kge_metrics']['post_ces']

                f.write("PRE-CES:\n")
                f.write(f"  Train MRR: {pre['train']['mrr']:.4f}\n")
                f.write(f"  Test MRR:  {pre['test']['mrr']:.4f}\n")
                f.write(f"  Gap:       {pre['gap']:.4f}\n\n")

                f.write("POST-CES:\n")
                f.write(f"  Train MRR: {post['train']['mrr']:.4f}\n")
                f.write(f"  Test MRR:  {post['test']['mrr']:.4f}\n")
                f.write(f"  Gap:       {post['gap']:.4f}\n")
                f.write(f"  Gap Improvement: {metrics['kge_metrics']['gap_improvement']:.4f}\n\n")
            else:
                f.write(f"\nTrain MRR: {metrics['kge_metrics']['final_train_mrr']:.4f}\n")
                f.write(f"Test MRR:  {metrics['kge_metrics']['final_test_mrr']:.4f}\n")
                f.write(f"Gap:       {metrics['kge_metrics']['final_train_test_gap']:.4f}\n\n")

            f.write("LEVEL 2: CLASSIFIER METRICS (Using POST-CES Embeddings)\n")
            f.write("-" * 60 + "\n")
            cv = metrics['cv_results']
            f.write(f"CV Accuracy: {cv['mean_accuracy']:.4f} ± {cv['std_accuracy']:.4f}\n")
            f.write(f"  95% CI: {cv['ci_accuracy'][0]:.4f}–{cv['ci_accuracy'][1]:.4f}\n")
            f.write(f"CV AUC:      {cv['mean_auc']:.4f} ± {cv['std_auc']:.4f}\n")
            f.write(f"  95% CI: {cv['ci_auc'][0]:.4f}–{cv['ci_auc'][1]:.4f}\n")
            f.write(f"CV LogLoss:  {cv['mean_logloss']:.4f} ± {cv['std_logloss']:.4f}\n")
            f.write(f"CV MPCE:     {cv['mean_mpce']:.4f} ± {cv['std_mpce']:.4f}\n\n")
            f.write(f"Test Accuracy: {metrics['test_accuracy']:.4f}\n")
            f.write(f"Test AUC:      {metrics['test_auc']:.4f}\n")
            f.write(f"Test LogLoss:  {metrics['test_logloss']:.4f}\n")
            f.write(f"Test MPCE:     {metrics['test_mpce']:.4f}\n\n")
            f.write(f"CV-Test Accuracy Gap: {abs(cv['mean_accuracy'] - metrics['test_accuracy']):.4f}\n")
            f.write(f"CV-Test AUC Gap:      {abs(cv['mean_auc'] - metrics['test_auc']):.4f}\n")

            if metrics['wilcoxon_p_value'] is not None:
                f.write(f"\nWilcoxon test p-value: {metrics['wilcoxon_p_value']:.4f}\n")

        joblib.dump(model, self.model_file)
        print(f"\nResults saved to: {self.results_file}")
        print(f"Statistics saved to: {self.statistical_file}")
        print(f"Model saved to: {self.model_file}")


# ==============================================================================
# MULTIPLE EXPERIMENTS WITH CES
# ==============================================================================

def run_one_model_all_repos(model: str, base_path: str = '../data',
                            search_type: str = 'smart',
                            apply_ces: bool = True,
                            ces_alpha: float = 0.5) -> Dict:
    """Ejecutar UN modelo en TODOS los repositorios con CES"""
    print("\n" + "=" * 80)
    print(f"RUNNING {model.upper()} ON ALL REPOSITORIES")
    print(f"Search type: {search_type}")
    print(f"CES: {'ENABLED' if apply_ces else 'DISABLED'} (alpha={ces_alpha})")
    print("=" * 80)

    results = {}

    for repo, dataset in REPOSITORIES.items():
        print(f"\n>>> Processing {repo.upper()} with {model.upper()}")

        kg_path = f"{base_path}/triples_raw/{repo}/{dataset}"
        emb_dir = f"{base_path}/c _091125/{repo}_ces_{model}"
        output_dir = f"{base_path}/model_output_091125/{repo}_ces_{model}"

        try:
            pipeline = IntegratedClassificationPipeline(
                kg_path=kg_path,
                emb_dir=emb_dir,
                output_dir=output_dir,
                model_type=model,
                search_type=search_type,
                n_folds=10,
                apply_ces=apply_ces,
                ces_alpha=ces_alpha
            )

            final_model, metrics = pipeline.run_classification_with_statistics(
                n_iter=20,
                cv=3
            )

            results[repo] = {
                'status': 'SUCCESS',
                'ces_applied': metrics['ces_applied'],
                'kge_metrics': metrics['kge_metrics'],
                'test_accuracy': metrics['test_accuracy'],
                'test_auc': metrics['test_auc'],
                'cv_mean_accuracy': metrics['cv_results']['mean_accuracy'],
                'cv_mean_auc': metrics['cv_results']['mean_auc'],
            }

            print(f"\n✓ {repo}/{model} completed successfully!")

        except Exception as e:
            print(f"\n✗ {repo}/{model} failed: {e}")
            import traceback
            traceback.print_exc()
            results[repo] = {'status': 'FAILED', 'error': str(e)}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f"{base_path}/model_output/{model.upper()}_ALL_REPOS_CES_SUMMARY_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 80)
    print(f"{model.upper()} ON ALL REPOSITORIES COMPLETED")
    print("=" * 80)
    print(f"Summary saved to: {summary_file}")

    return results


def run_all_models_all_repos(models_to_run: List[str] = MODELS_AVAILABLE,
                             base_path: str = '../data',
                             search_type: str = 'smart',
                             apply_ces: bool = True,
                             ces_alpha: float = 0.5) -> Dict:
    """Ejecutar TODOS los modelos en TODOS los repositorios con CES"""
    print("\n" + "=" * 80)
    print("RUNNING ALL MODELS ON ALL REPOSITORIES WITH CES")
    print(f"Models: {models_to_run}")
    print(f"Search type: {search_type}")
    print(f"CES: {'ENABLED' if apply_ces else 'DISABLED'} (alpha={ces_alpha})")
    print("=" * 80)

    all_results = {}

    for repo, dataset in REPOSITORIES.items():
        print(f"\n{'=' * 80}")
        print(f"PROCESSING REPOSITORY: {repo.upper()}")
        print(f"{'=' * 80}")

        kg_path = f"{base_path}/triples_raw/{repo}/{dataset}"
        all_results[repo] = {}

        for model in models_to_run:
            print(f"\n>>> Training {model.upper()} on {repo.upper()}")

            emb_dir = f"{base_path}/triples_emb_091125/{repo}_ces_{model}"
            output_dir = f"{base_path}/model_output_091125/{repo}_ces_{model}"

            try:
                pipeline = IntegratedClassificationPipeline(
                    kg_path=kg_path,
                    emb_dir=emb_dir,
                    output_dir=output_dir,
                    model_type=model,
                    search_type=search_type,
                    n_folds=10,
                    apply_ces=apply_ces,
                    ces_alpha=ces_alpha
                )

                final_model, metrics = pipeline.run_classification_with_statistics(
                    n_iter=20,
                    cv=3
                )

                all_results[repo][model] = {
                    'status': 'SUCCESS',
                    'ces_applied': metrics['ces_applied'],
                    'kge_metrics': metrics['kge_metrics'],
                    'test_accuracy': metrics['test_accuracy'],
                    'test_auc': metrics['test_auc'],
                    'cv_mean_accuracy': metrics['cv_results']['mean_accuracy'],
                    'cv_mean_auc': metrics['cv_results']['mean_auc'],
                }

                print(f"\n✓ {repo}/{model} completed successfully!")

            except Exception as e:
                print(f"\n✗ {repo}/{model} failed: {e}")
                import traceback
                traceback.print_exc()
                all_results[repo][model] = {'status': 'FAILED', 'error': str(e)}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f"{base_path}/model_output/COMPLETE_CES_EXPERIMENTAL_SUMMARY_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED")
    print("=" * 80)
    print(f"Summary saved to: {summary_file}")

    # Print summary table
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    for repo in all_results:
        print(f"\n{repo.upper()}:")
        for model in all_results[repo]:
            status = all_results[repo][model]['status']
            if status == 'SUCCESS':
                acc = all_results[repo][model]['test_accuracy']
                auc = all_results[repo][model]['test_auc']
                ces = "✓ CES" if all_results[repo][model]['ces_applied'] else ""
                print(f"  {model}: Acc={acc:.4f}, AUC={auc:.4f} {ces}")
            else:
                print(f"  {model}: FAILED ✗")

    return all_results


# ==============================================================================
# MAIN CON OPCIONES CES
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("INTEGRATED KGE + CLASSIFICATION WITH CES STRATEGY")
    print("=" * 80)
    print("\nSelect execution mode:")
    print("1. Single model + single repository")
    print("2. Single model + ALL repositories")
    print("3. ALL models + ALL repositories")
    print("4. Exit")

    choice = input("\nEnter choice (1-4): ").strip()

    if choice == '4':
        print("Exiting...")
        sys.exit(0)

    # Ask for search type
    print("\nSelect hyperparameter search type:")
    print("  smart: Smart search (model-specific, ADJUSTED, faster)")
    print("  grid:  Grid search (exhaustive, slower)")
    search_type = input("Enter search type (smart/grid, default=smart): ").strip().lower() or 'smart'

    if search_type not in ['smart', 'grid']:
        print(f"Invalid search type: {search_type}. Using 'smart'.")
        search_type = 'smart'

    # Ask for CES
    print("\nApply CES (Capacity-aware Entity Scaling)?")
    print("  Recommended: YES for models with high train-test gap")
    apply_ces_input = input("Apply CES? (y/n, default=y): ").strip().lower() or 'y'
    apply_ces = (apply_ces_input == 'y')

    ces_alpha = 0.5
    if apply_ces:
        print("\nSelect CES alpha (strength of rescaling):")
        print("  0.3 = Soft rescaling")
        print("  0.5 = Moderate rescaling (RECOMMENDED)")
        print("  0.7 = Strong rescaling")
        print("  1.0 = Very strong rescaling")
        ces_alpha_input = input("Enter alpha (default=0.5): ").strip()
        if ces_alpha_input:
            try:
                ces_alpha = float(ces_alpha_input)
                if ces_alpha <= 0 or ces_alpha > 1:
                    print("Invalid alpha. Using 0.5")
                    ces_alpha = 0.5
            except:
                print("Invalid input. Using 0.5")
                ces_alpha = 0.5

    if choice == '1':
        print("\nAvailable repositories:", list(REPOSITORIES.keys()))
        repo = input("Enter repository: ").strip().lower()

        print("\nAvailable models:", MODELS_AVAILABLE)
        model = input("Enter model: ").strip().lower()

        if repo not in REPOSITORIES:
            print(f"Invalid repository: {repo}")
            sys.exit(1)

        if model not in MODELS_AVAILABLE:
            print(f"Invalid model: {model}")
            sys.exit(1)

        if 'reuters' in repo:
            repo = 'reuters_activities'

        kg_path = f"../data/triples_raw/{repo}/{REPOSITORIES[repo]}"
        emb_dir = f"../data/triples_emb_091125/{repo}_ces_{model}"
        output_dir = f"../data/model_output_091125/{repo}_ces_{model}"

        pipeline = IntegratedClassificationPipeline(
            kg_path=kg_path,
            emb_dir=emb_dir,
            output_dir=output_dir,
            model_type=model,
            search_type=search_type,
            n_folds=10,
            apply_ces=apply_ces,
            ces_alpha=ces_alpha
        )

        final_model, metrics = pipeline.run_classification_with_statistics(
            n_iter=20,
            cv=3
        )

        print(f"\n✓ Pipeline completed!")
        print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
        print(f"Test AUC: {metrics['test_auc']:.4f}")
        if metrics.get('ces_applied'):
            print(f"CES Gap Improvement: {metrics['kge_metrics']['gap_improvement']:.4f}")

    elif choice == '2':
        print("\nAvailable models:", MODELS_AVAILABLE)
        model = input("Enter model: ").strip().lower()

        if model not in MODELS_AVAILABLE:
            print(f"Invalid model: {model}")
            sys.exit(1)

        print(f"\nRunning {model.upper()} on ALL repositories with CES...")
        run_one_model_all_repos(
            model=model,
            base_path='../data',
            search_type=search_type,
            apply_ces=apply_ces,
            ces_alpha=ces_alpha
        )

    elif choice == '3':
        print("\nRunning ALL models on ALL repositories with CES...")
        print("This will take considerable time...")

        confirm = input("Continue? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Cancelled.")
            sys.exit(0)

        run_all_models_all_repos(
            models_to_run=MODELS_AVAILABLE,
            base_path='../data',
            search_type=search_type,
            apply_ces=apply_ces,
            ces_alpha=ces_alpha
        )

    else:
        print("Invalid choice. Exiting...")
        sys.exit(1)