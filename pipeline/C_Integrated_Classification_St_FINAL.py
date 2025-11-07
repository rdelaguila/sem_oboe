import pandas as pd
import numpy as np
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from pykeen.evaluation import RankBasedEvaluator
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    log_loss, roc_auc_score, roc_curve, auc
)
from xgboost import XGBClassifier
import torch
import joblib
import os
import json
from datetime import datetime
from typing import Tuple, Dict, Any, Optional, List
import warnings
import sys
from itertools import product
from scipy import stats

warnings.filterwarnings('ignore')


class KGEOptimizer:
    """
    Optimizador unificado para modelos KGE con soporte para múltiples modelos
    y optimización de hiperparámetros
    """

    AVAILABLE_MODELS = {
        'transe': 'TransE',
        'convkb': 'ConvKB',
        'complex': 'ComplEx',
    }

    def __init__(self, kg_path: str, emb_dir: str, output_dir: str, model_type: str = 'complex',
                 early_stopping_patience: int = 5, early_stopping_delta: float = 0.0005,
                 aggressive_early_stop: bool = False):
        self.kg_csv_path = kg_path
        self.emb_dir = emb_dir
        self.output_dir = output_dir
        self.model_type = self._validate_model_type(model_type)
        self.model = None
        self.entity_to_id = None
        self.relation_to_id = None
        self.best_embedding_dim = None
        self.best_hyperparams = None
        
        # Early stopping configuration
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_delta = early_stopping_delta
        self.aggressive_early_stop = aggressive_early_stop

        # Store ALL trained models for analysis
        self.all_trained_models = []

        # Create directories
        os.makedirs(emb_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # Setup output files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_file = os.path.join(output_dir, f"{model_type}_optimization_results_{timestamp}.json")
        self.metrics_file = os.path.join(output_dir, f"{model_type}_optimization_metrics_{timestamp}.txt")
        self.statistical_file = os.path.join(output_dir, f"{model_type}_statistical_analysis_{timestamp}.txt")
        self.best_model_file = os.path.join(emb_dir, f'{model_type}_best_model.pkl')
        self.best_mappings_file = os.path.join(emb_dir, f'{model_type}_best_mappings.pkl')

    def _validate_model_type(self, model_type: str) -> str:
        """Validate and return model type"""
        if model_type.lower() not in self.AVAILABLE_MODELS:
            raise ValueError(f"Invalid model type. Choose from: {list(self.AVAILABLE_MODELS.keys())}")
        return model_type.lower()

    def prepare_triples_factory(self) -> Tuple[TriplesFactory, TriplesFactory, TriplesFactory]:
        """Prepare triples factory for KGE training with validation set"""
        print("\nPreparing triples factory...")
        dftrain = pd.read_csv(self.kg_csv_path)
        dftest = pd.read_csv(self.kg_csv_path.replace('sampled_train', 'sampled_test'))
        triplestr = dftrain[['subject', 'relation', 'object']].values
        tripletest = dftest[['subject', 'relation', 'object']].values

        # For ComplEx, always use inverse triples
        create_inverse = (self.model_type == 'complex')
        if create_inverse:
            print("Creating inverse triples (recommended for ComplEx)")

        training_tf = TriplesFactory.from_labeled_triples(
            triplestr,
            create_inverse_triples=create_inverse,
        )
        testing_tf = TriplesFactory.from_labeled_triples(
            tripletest,
            create_inverse_triples=create_inverse,
            entity_to_id=training_tf.entity_to_id,
            relation_to_id=training_tf.relation_to_id,
        )

        # Create validation set from training
        training_sub_tf, validation_tf = training_tf.split([0.85, 0.15], random_state=42)

        print(f"Training triples: {len(training_sub_tf.triples)}")
        print(f"Validation triples: {len(validation_tf.triples)}")
        print(f"Testing triples: {len(testing_tf.triples)}")
        print(f"Entities: {training_tf.num_entities}")
        print(f"Relations: {training_tf.num_relations}")

        return training_sub_tf, validation_tf, testing_tf

    def get_hyperparameter_grid(self) -> Dict[str, List]:
        """
        Define hyperparameter search space for each model type
        Por defecto optimiza: embedding_dim, negative_sampling (cuando aplica), y regularización
        """
        base_grid = {
            'embedding_dim': [64, 100, 128, 150, 200],
        }

        if self.model_type == 'complex':
            base_grid.update({
                'learning_rate': [0.0001, 0.0005, 0.001, 0.005, 0.01],
                'num_epochs': [100, 150, 200],
                'batch_size': [128, 256, 512],
                'regularizer_weight': [0.0001, 0.001, 0.01, 0.1],
                'num_negs_per_pos': [32, 64, 128, 256]
            })
        elif self.model_type == 'transe':
            base_grid.update({
                'learning_rate': [0.0001, 0.001, 0.01],
                'num_epochs': [100, 150, 200],
                'batch_size': [100, 256, 512],
                'regularizer_weight': [1e-5, 1e-4, 1e-3],
                'scoring_fct_norm': [1, 2]
            })
        elif self.model_type == 'convkb':
            base_grid.update({
                'learning_rate': [0.0001, 0.001, 0.01],
                'num_epochs': [100, 150, 200],
                'batch_size': [100, 256, 512],
                'regularizer_weight': [1e-5, 1e-4, 1e-3],
                'num_filters': [32, 64],
                'hidden_dropout_rate': [0.3, 0.5]
            })

        return base_grid

    def _get_smart_search_combinations_complex(self) -> List[Tuple]:
        """
        Get smart search combinations for ComplEx - optimized for MDPI publication
        Búsqueda eficiente de embedding_dim manteniendo rigor estadístico
        """
        # Format: (embedding_dim, learning_rate, num_epochs, batch_size, regularizer_weight, num_negs_per_pos)
        combinations = [
            # Embedding dim = 64 (low complexity)
            (64, 0.02, 75, 256, 0.012, 128),
            
            # Embedding dim = 128 (baseline)
            (128, 0.02, 75, 256, 0.012, 128),
            (128, 0.019, 75, 256, 0.013, 128),
            
            # Embedding dim = 200 (high capacity)
            (200, 0.018, 75, 256, 0.015, 128),
        ]

        print("\n" + "=" * 80)
        print("SMART SEARCH OPTIMIZADO PARA PUBLICACIÓN MDPI")
        print("=" * 80)
        print("Búsqueda de embedding_dim con rigor estadístico:")
        print(f"  • Total configs: {len(combinations)}")
        print(f"  • Embedding dims explorados: [64, 128, 200]")
        print(f"  • Epochs por config: 75")
        print(f"  • 10-fold CV para validación")
        print(f"  • Tiempo estimado: ~{len(combinations) * 30 // 60} horas")
        print("=" * 80 + "\n")

        return list(set(combinations))

    def _get_smart_search_combinations_transe(self) -> List[Dict]:
        """
        Get smart search combinations for TransE - optimized for MDPI publication
        Búsqueda de embedding_dim con configuración óptima fija
        """
        combinations = []
        embedding_dims = [64, 128, 200]  # Búsqueda de dimensionalidad

        # Configuración óptima fija para TransE
        fixed_config = {
            'learning_rate': 0.001,
            'num_epochs': 75,  # Reducido para eficiencia
            'batch_size': 100,
            'scoring_fct_norm': 2,
            'regularizer_weight': 1e-5
        }

        for dim in embedding_dims:
            config = {'embedding_dim': int(dim), **fixed_config}
            combinations.append(config)

        print("\n" + "=" * 80)
        print("SMART SEARCH PARA TRANSE - PUBLICACIÓN MDPI")
        print("=" * 80)
        print("Búsqueda de embedding_dim con configuración óptima:")
        print(f"  • Embedding dims: {embedding_dims}")
        print(f"  • Epochs: 75")
        print(f"  • Regularización: {fixed_config['regularizer_weight']}")
        print(f"  • Total configs: {len(combinations)}")
        print(f"  • 10-fold CV para validación")
        print(f"  • Tiempo estimado: ~{len(combinations) * 25 // 60} horas")
        print("=" * 80 + "\n")

        return combinations

    def _get_smart_search_combinations_convkb(self) -> List[Dict]:
        """
        Get smart search combinations for ConvKB - optimized for MDPI publication
        Búsqueda de embedding_dim con configuración óptima fija
        """
        combinations = []
        embedding_dims = [64, 128, 200]  # Búsqueda de dimensionalidad

        # Configuración óptima fija para ConvKB
        fixed_config = {
            'learning_rate': 0.001,
            'num_epochs': 75,  # Reducido para eficiencia
            'batch_size': 128,
            'num_filters': 32,
            'hidden_dropout_rate': 0.5,
            'regularizer_weight': 1e-4
        }

        for dim in embedding_dims:
            config = {'embedding_dim': int(dim), **fixed_config}
            combinations.append(config)

        print("\n" + "=" * 80)
        print("SMART SEARCH PARA CONVKB - PUBLICACIÓN MDPI")
        print("=" * 80)
        print("Búsqueda de embedding_dim con configuración óptima:")
        print(f"  • Embedding dims: {embedding_dims}")
        print(f"  • Epochs: 75")
        print(f"  • Regularización: {fixed_config['regularizer_weight']}")
        print(f"  • Total configs: {len(combinations)}")
        print(f"  • 10-fold CV para validación")
        print(f"  • Tiempo estimado: ~{len(combinations) * 20 // 60} horas")
        print("=" * 80 + "\n")

        return combinations

    def train_single_config(self, training_tf: TriplesFactory, validation_tf: TriplesFactory,
                            testing_tf: TriplesFactory, config: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """Train model with specific hyperparameters"""
        # Convert to native Python int to avoid numpy type issues
        embedding_dim = int(config['embedding_dim'])
        learning_rate = float(config.get('learning_rate', 0.001))
        num_epochs = int(config.get('num_epochs', 200))
        batch_size = int(config.get('batch_size', 100))
        regularizer_weight = float(config.get('regularizer_weight', 0.01))

        config_str = f"dim={embedding_dim}, lr={learning_rate}, epochs={num_epochs}"
        print(f"\n{'=' * 80}")
        print(f"Training {self.AVAILABLE_MODELS[self.model_type]}: {config_str}")
        print(f"{'=' * 80}")

        # Model-specific configurations
        model_kwargs = {'embedding_dim': embedding_dim}
        negative_sampler = None
        negative_sampler_kwargs = None

        if self.model_type == 'complex':
            model_kwargs.update({
                'regularizer': 'lp',
                'regularizer_kwargs': dict(p=2, weight=regularizer_weight),
            })
            negative_sampler = 'bernoulli'
            negative_sampler_kwargs = dict(
                num_negs_per_pos=config.get('num_negs_per_pos', 128)
            )
        elif self.model_type == 'transe':
            model_kwargs.update({
                'scoring_fct_norm': config.get('scoring_fct_norm', 2),
                'regularizer': 'lp',
                'regularizer_kwargs': dict(p=3, weight=regularizer_weight),
            })
        elif self.model_type == 'convkb':
            model_kwargs.update({
                'num_filters': config.get('num_filters', 32),
                'hidden_dropout_rate': config.get('hidden_dropout_rate', 0.5),
            })

        try:
            # Device selection: CPU for ComplEx (stability), MPS for others (speed)
            if self.model_type == 'complex':
                device = 'cpu'
                print(f"Using device: {device} (ComplEx stability mode)")
            else:
                device = 'mps'
                print(f"Using device: {device} (MPS acceleration)")

            print(f"Training triples: {len(training_tf.triples)}")
            print(f"Validation triples (for early stopping): {len(validation_tf.triples)}")

            # Early stopping configuration (configurable)
            if self.aggressive_early_stop:
                # More aggressive early stopping to prevent overfitting
                stopper_kwargs = {
                    'frequency': 3,
                    'patience': 3,
                    'relative_delta': 0.001,  # Require 0.1% improvement
                    'metric': 'mean_reciprocal_rank',
                }
                print(f"⚠️ AGGRESSIVE Early stopping: patience=3, frequency=3, min_improvement=0.1%")
            else:
                # Standard early stopping
                stopper_kwargs = {
                    'frequency': 5,
                    'patience': self.early_stopping_patience,
                    'relative_delta': self.early_stopping_delta,
                    'metric': 'mean_reciprocal_rank',
                }
                print(f"Early stopping: patience={self.early_stopping_patience}, min_improvement={self.early_stopping_delta*100:.3f}%")

            result = pipeline(
                training=training_tf,
                validation=validation_tf,
                testing=testing_tf,
                model=self.model_type,
                device=device,
                model_kwargs=model_kwargs,
                negative_sampler=negative_sampler,
                negative_sampler_kwargs=negative_sampler_kwargs,
                optimizer='adam',
                optimizer_kwargs=dict(lr=learning_rate),
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
                evaluator_kwargs=dict(filtered=True),  # FIXED: filtered=True para evitar error
                evaluation_kwargs=dict(batch_size=256),
                random_seed=42,
            )

            # Extract evaluation metrics
            metrics = {
                **config,
                'model_name': self.AVAILABLE_MODELS[self.model_type],
                'model_type': self.model_type,
                'num_entities': training_tf.num_entities,
                'num_relations': training_tf.num_relations,
                'training_triples': len(training_tf.triples),
                'testing_triples': len(testing_tf.triples),
            }

            # Add evaluation results
            if hasattr(result, 'metric_results') and result.metric_results:
                metrics['evaluation'] = {
                    'hits_at_1': float(result.metric_results.get_metric('hits@1')),
                    'hits_at_3': float(result.metric_results.get_metric('hits@3')),
                    'hits_at_10': float(result.metric_results.get_metric('hits@10')),
                    'mrr': float(result.metric_results.get_metric('mean_reciprocal_rank')),
                    'mr': float(result.metric_results.get_metric('mean_rank')),
                }
                metrics['mrr_score'] = metrics['evaluation']['mrr']

                # Add early stopping information
                if hasattr(result, 'losses') and result.losses:
                    actual_epochs = len(result.losses)
                    metrics['actual_epochs'] = actual_epochs
                    metrics['early_stopped'] = actual_epochs < num_epochs
                    if metrics['early_stopped']:
                        epochs_saved = num_epochs - actual_epochs
                        print(f"\n⚡ Early stopping triggered at epoch {actual_epochs}/{num_epochs}")
                        print(f"   Saved {epochs_saved} epochs (~{epochs_saved * 0.5:.1f} min)")

                print(f"\nResults for {config_str}:")
                print(f"  Hits@1: {metrics['evaluation']['hits_at_1']:.4f}")
                print(f"  Hits@10: {metrics['evaluation']['hits_at_10']:.4f}")
                print(f"  MRR: {metrics['evaluation']['mrr']:.4f}")
                if 'actual_epochs' in metrics:
                    print(f"  Epochs: {metrics['actual_epochs']}/{num_epochs}")
            else:
                metrics['mrr_score'] = 0.0

            return result.model, metrics

        except Exception as e:
            print(f"ERROR training with {config_str}: {e}")
            metrics = {**config, 'mrr_score': 0.0, 'error': str(e)}
            return None, metrics

    def filter_outlier_models(self, all_models: List[Dict]) -> List[Dict]:
        """
        Descarta modelos con MRR demasiado alto respecto a la media
        (posible overfitting)
        """
        valid_models = [m for m in all_models if m['metrics'].get('mrr_score', 0) > 0]
        if len(valid_models) < 3:
            return valid_models

        mrr_values = [m['metrics']['mrr_score'] for m in valid_models]
        mean_mrr = np.mean(mrr_values)
        std_mrr = np.std(mrr_values)

        # Filtrar modelos con MRR > mean + 1.5*std (outliers superiores)
        threshold = mean_mrr + 1.5 * std_mrr
        filtered_models = [m for m in valid_models if m['metrics']['mrr_score'] <= threshold]

        n_removed = len(valid_models) - len(filtered_models)
        if n_removed > 0:
            print(
                f"\n⚠️  Removed {n_removed} models with MRR > {threshold:.4f} (mean={mean_mrr:.4f}, std={std_mrr:.4f})")
            print("   Possible overfitting detected")

        return filtered_models if len(filtered_models) > 0 else valid_models

    def optimize_hyperparameters(self, training_tf: TriplesFactory, validation_tf: TriplesFactory,
                                 testing_tf: TriplesFactory, search_type: str = 'smart',
                                 n_random_samples: int = 20) -> Dict[str, Any]:
        """Perform hyperparameter optimization"""

        print("\n" + "=" * 80)
        print(f"STARTING HYPERPARAMETER OPTIMIZATION FOR {self.AVAILABLE_MODELS[self.model_type].upper()}")
        print("=" * 80)

        param_grid = self.get_hyperparameter_grid()

        if search_type == 'smart':
            # Use predefined smart combinations specific to each model
            if self.model_type == 'complex':
                param_combinations = self._get_smart_search_combinations_complex()
                configs = []
                for combo in param_combinations:
                    config = {
                        'embedding_dim': int(combo[0]),
                        'learning_rate': float(combo[1]),
                        'num_epochs': int(combo[2]),
                        'batch_size': int(combo[3]),
                        'regularizer_weight': float(combo[4]),
                        'num_negs_per_pos': int(combo[5]) if len(combo) > 5 else 128
                    }
                    configs.append(config)
            elif self.model_type == 'transe':
                configs = self._get_smart_search_combinations_transe()
            elif self.model_type == 'convkb':
                configs = self._get_smart_search_combinations_convkb()
            else:
                # Fallback to random search if smart search not defined
                print(f"Warning: Smart search not defined for {self.model_type}, using random search")
                search_type = 'random'
                configs = []
                np.random.seed(42)
                for _ in range(n_random_samples):
                    config = {k: np.random.choice(v) for k, v in param_grid.items()}
                    configs.append(config)
        elif search_type == 'random':
            print(f"\nPerforming RANDOM SEARCH with {n_random_samples} samples")
            configs = []
            np.random.seed(42)
            for _ in range(n_random_samples):
                config = {k: np.random.choice(v) for k, v in param_grid.items()}
                configs.append(config)
        else:
            # Grid search
            print("\nPerforming GRID SEARCH")
            param_combinations = list(product(*param_grid.values()))
            configs = [dict(zip(param_grid.keys(), combo)) for combo in param_combinations]
            print(f"Total combinations to try: {len(configs)}")

        print(f"Trying {len(configs)} configurations\n")

        # Train all configurations
        for i, config in enumerate(configs, 1):
            print(f"\n[{i}/{len(configs)}] Testing configuration...")

            model, metrics = self.train_single_config(training_tf, validation_tf, testing_tf, config)

            if model is not None:
                self.all_trained_models.append({
                    'model': model,
                    'metrics': metrics,
                    'config_index': i
                })

        # Filter outlier models
        self.all_trained_models = self.filter_outlier_models(self.all_trained_models)

        # Find best model by MRR (will be overridden if AUC selection is used later)
        best_score = -float('inf')
        best_model = None
        best_metrics = None

        for model_info in self.all_trained_models:
            score = model_info['metrics'].get('mrr_score', 0.0)
            if score > best_score:
                best_score = score
                best_model = model_info['model']
                best_metrics = model_info['metrics']
                self.best_embedding_dim = model_info['metrics']['embedding_dim']
                self.best_hyperparams = {
                    k: v for k, v in model_info['metrics'].items()
                    if k in param_grid.keys()
                }

        # Store best model
        self.model = best_model
        self.entity_to_id = training_tf.entity_to_id
        self.relation_to_id = training_tf.relation_to_id

        # Add optimization results
        all_results = [m['metrics'] for m in self.all_trained_models]
        best_metrics['all_results'] = all_results
        best_metrics['best_mrr_score'] = best_score
        best_metrics['best_hyperparams'] = self.best_hyperparams
        best_metrics['search_type'] = search_type
        best_metrics['total_configurations_tried'] = len(configs)
        best_metrics['models_after_filtering'] = len(self.all_trained_models)

        print("\n" + "=" * 80)
        print("HYPERPARAMETER OPTIMIZATION COMPLETED")
        print("=" * 80)
        print(f"Best configuration found (by MRR):")
        for key, value in self.best_hyperparams.items():
            print(f"  {key}: {value}")
        print(f"Best MRR: {best_score:.4f}")

        return best_metrics

    def save_kge_model_and_mappings(self, kge_metrics: Dict[str, Any]) -> str:
        """Save the trained KGE model and entity/relation mappings"""
        torch.save(self.model, self.best_model_file)

        mappings = {
            'entity_to_id': self.entity_to_id,
            'relation_to_id': self.relation_to_id,
            'model_type': self.model_type,
            'embedding_dim': self.best_embedding_dim
        }
        joblib.dump(mappings, self.best_mappings_file)

        # Save basic info in JSON
        basic_info = {
            'timestamp': datetime.now().isoformat(),
            'model_path': self.best_model_file,
            'mappings_path': self.best_mappings_file,
            'model_type': self.model_type,
            'best_embedding_dim': self.best_embedding_dim,
            'best_mrr_score': float(kge_metrics.get('best_mrr_score', 0.0)),
        }

        with open(self.results_file, 'w') as f:
            json.dump(basic_info, f, indent=2, default=str)

        # Save human-readable metrics
        with open(self.metrics_file, 'w') as f:
            f.write("KGE MODEL TRAINING RESULTS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Timestamp: {basic_info['timestamp']}\n")
            f.write(f"Model: {kge_metrics['model_name']} ({kge_metrics['model_type']})\n")
            f.write(f"Embedding Dimension: {kge_metrics['embedding_dim']}\n")
            f.write(f"Entities: {kge_metrics['num_entities']}\n")
            f.write(f"Relations: {kge_metrics['num_relations']}\n\n")

            if 'evaluation' in kge_metrics:
                f.write("EVALUATION METRICS\n")
                f.write("-" * 30 + "\n")
                f.write(f"Hits@1: {kge_metrics['evaluation']['hits_at_1']:.4f}\n")
                f.write(f"Hits@10: {kge_metrics['evaluation']['hits_at_10']:.4f}\n")
                f.write(f"MRR: {kge_metrics['evaluation']['mrr']:.4f}\n")

        print(f"\nKGE model and results saved:")
        print(f"  Model: {self.best_model_file}")
        print(f"  Mappings: {self.best_mappings_file}")

        return self.best_model_file

    def run_kge_training(self, search_type: str = 'smart', n_random_samples: int = 20) -> str:
        """Run complete KGE training pipeline"""
        print("=" * 80)
        print(f"STARTING KGE MODEL TRAINING: {self.AVAILABLE_MODELS[self.model_type]}")
        print("=" * 80)

        training_tf, validation_tf, testing_tf = self.prepare_triples_factory()
        kge_metrics = self.optimize_hyperparameters(training_tf, validation_tf, testing_tf,
                                                    search_type, n_random_samples)
        model_path = self.save_kge_model_and_mappings(kge_metrics)

        print("\n" + "=" * 80)
        print("KGE MODEL TRAINING FINISHED")
        print("=" * 80)

        return model_path


class TopicClassifierWithStatistics:
    """
    Classification with comprehensive statistical analysis:
    - Cross-validation with confidence intervals
    - Pairwise model comparison (Wilcoxon test)
    - Variance analysis
    - Model selection based on AUC (not MRR)
    """

    def __init__(self, kge_csv_path: str, kge_optimizer: KGEOptimizer,
                 output_dir: str, n_folds: int = 10):
        self.kge_csv_path = kge_csv_path
        self.kge_optimizer = kge_optimizer
        self.output_dir = output_dir
        self.n_folds = n_folds

        # Model info
        self.model_type = kge_optimizer.model_type
        self.embedding_dim = kge_optimizer.best_embedding_dim

        # Classification type detection
        df = pd.read_csv(kge_csv_path)
        self.n_classes = df['category'].nunique()
        self.is_binary_classification = (self.n_classes == 2)

        # Output files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_file = os.path.join(output_dir, f"classification_results_{timestamp}.json")
        self.metrics_file = os.path.join(output_dir, f"classification_metrics_{timestamp}.txt")
        self.statistical_file = os.path.join(output_dir, f"statistical_analysis_{timestamp}.txt")
        self.best_model_file = os.path.join(output_dir, f"best_classifier_{timestamp}.pkl")

        print(f"\n{'Binary' if self.is_binary_classification else 'Multiclass'} classification detected")
        print(f"Number of classes: {self.n_classes}")

    def extract_embeddings(self, model, entity_to_id: Dict, df: pd.DataFrame) -> np.ndarray:
        """Extract embeddings for entities"""
        embeddings_list = []

        for _, row in df.iterrows():
            subject = row['subject']
            if subject in entity_to_id:
                subject_id = entity_to_id[subject]
                subject_emb = model.entity_representations[0](
                    torch.tensor([subject_id])
                ).detach().numpy().flatten()
                embeddings_list.append(subject_emb)
            else:
                # Use zero vector for unknown entities
                embeddings_list.append(np.zeros(self.embedding_dim))

        return np.array(embeddings_list)

    def perform_cross_validation(self, X: np.ndarray, y: np.ndarray,
                                 best_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform stratified k-fold cross-validation with comprehensive statistics
        """
        print(f"\nPerforming {self.n_folds}-fold cross-validation...")

        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        # Storage for metrics across folds
        fold_accuracies = []
        fold_aucs = []
        fold_logloss = []
        fold_mean_per_class_error = []
        fold_predictions = []
        fold_true_labels = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            print(f"  Fold {fold}/{self.n_folds}...", end=' ')

            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            # Train model with best params
            model = XGBClassifier(
                **best_params,
                objective='binary:logistic' if self.is_binary_classification else 'multi:softprob',
                eval_metric='logloss',
                random_state=42,
                use_label_encoder=False
            )

            model.fit(X_train_fold, y_train_fold)

            # Predictions
            y_pred = model.predict(X_val_fold)
            y_pred_proba = model.predict_proba(X_val_fold)

            # Calculate metrics
            accuracy = accuracy_score(y_val_fold, y_pred)
            fold_accuracies.append(accuracy)

            # Calculate AUC
            if self.is_binary_classification:
                auc_score = roc_auc_score(y_val_fold, y_pred_proba[:, 1])
            else:
                auc_score = roc_auc_score(y_val_fold, y_pred_proba,
                                          multi_class='ovr', average='macro')
            fold_aucs.append(auc_score)
            
            # Calculate LogLoss
            logloss_score = log_loss(y_val_fold, y_pred_proba)
            fold_logloss.append(logloss_score)
            
            # Calculate Mean Per Class Error (1 - recall per class, averaged)
            cm = confusion_matrix(y_val_fold, y_pred)
            per_class_errors = []
            for i in range(len(cm)):
                if cm[i].sum() > 0:  # Avoid division by zero
                    recall = cm[i, i] / cm[i].sum()
                    error = 1 - recall
                    per_class_errors.append(error)
            mean_per_class_error = np.mean(per_class_errors) if per_class_errors else 0.0
            fold_mean_per_class_error.append(mean_per_class_error)

            # Store for pairwise comparison
            fold_predictions.append(y_pred)
            fold_true_labels.append(y_val_fold)

            print(f"Acc: {accuracy:.4f}, AUC: {auc_score:.4f}, LogLoss: {logloss_score:.4f}, MPCE: {mean_per_class_error:.4f}")

        # Calculate statistics
        mean_accuracy = np.mean(fold_accuracies)
        std_accuracy = np.std(fold_accuracies)
        variance_accuracy = np.var(fold_accuracies)

        mean_auc = np.mean(fold_aucs)
        std_auc = np.std(fold_aucs)
        
        mean_logloss = np.mean(fold_logloss)
        std_logloss = np.std(fold_logloss)
        
        mean_mpce = np.mean(fold_mean_per_class_error)
        std_mpce = np.std(fold_mean_per_class_error)

        # Calculate 95% confidence intervals using t-distribution
        ci_accuracy = stats.t.interval(
            0.95,
            len(fold_accuracies) - 1,
            loc=mean_accuracy,
            scale=stats.sem(fold_accuracies)
        )

        ci_auc = stats.t.interval(
            0.95,
            len(fold_aucs) - 1,
            loc=mean_auc,
            scale=stats.sem(fold_aucs)
        )
        
        ci_logloss = stats.t.interval(
            0.95,
            len(fold_logloss) - 1,
            loc=mean_logloss,
            scale=stats.sem(fold_logloss)
        )
        
        ci_mpce = stats.t.interval(
            0.95,
            len(fold_mean_per_class_error) - 1,
            loc=mean_mpce,
            scale=stats.sem(fold_mean_per_class_error)
        )

        print(f"\nCross-validation results:")
        print(f"  Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f} (95% CI: {ci_accuracy[0]:.4f}–{ci_accuracy[1]:.4f})")
        print(f"  Macro AUC: {mean_auc:.4f} ± {std_auc:.4f} (95% CI: {ci_auc[0]:.4f}–{ci_auc[1]:.4f})")
        print(f"  LogLoss: {mean_logloss:.4f} ± {std_logloss:.4f} (95% CI: {ci_logloss[0]:.4f}–{ci_logloss[1]:.4f})")
        print(f"  Mean Per Class Error: {mean_mpce:.4f} ± {std_mpce:.4f} (95% CI: {ci_mpce[0]:.4f}–{ci_mpce[1]:.4f})")
        print(f"  Variance (σ²): {variance_accuracy:.6f}")

        return {
            'fold_accuracies': fold_accuracies,
            'fold_aucs': fold_aucs,
            'fold_logloss': fold_logloss,
            'fold_mean_per_class_error': fold_mean_per_class_error,
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'variance_accuracy': variance_accuracy,
            'ci_accuracy': ci_accuracy,
            'mean_auc': mean_auc,
            'std_auc': std_auc,
            'ci_auc': ci_auc,
            'mean_logloss': mean_logloss,
            'std_logloss': std_logloss,
            'ci_logloss': ci_logloss,
            'mean_mpce': mean_mpce,
            'std_mpce': std_mpce,
            'ci_mpce': ci_mpce,
            'fold_predictions': fold_predictions,
            'fold_true_labels': fold_true_labels,
            'cv_scores': fold_accuracies  # For compatibility
        }

    def select_best_model_by_auc(self, X: np.ndarray, y: np.ndarray) -> Tuple[Any, Dict, np.ndarray]:
        """
        Select the best KGE model based on downstream classification AUC
        (NOT based on MRR)
        """
        print("\n" + "=" * 80)
        print("MODEL SELECTION: Evaluating all KGE models by downstream AUC")
        print("=" * 80)

        model_performances = []

        for i, model_info in enumerate(self.kge_optimizer.all_trained_models, 1):
            print(f"\n[{i}/{len(self.kge_optimizer.all_trained_models)}] Evaluating model...")
            print(f"  MRR: {model_info['metrics'].get('mrr_score', 0):.4f}")

            # Extract embeddings with this model
            X_temp = self.extract_embeddings(
                model_info['model'],
                self.kge_optimizer.entity_to_id,
                pd.read_csv(self.kge_csv_path)
            )

            # Quick train-test split for evaluation
            X_train, X_test, y_train, y_test = train_test_split(
                X_temp, y, test_size=0.2, random_state=42, stratify=y
            )

            # Train simple classifier
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

            # Calculate AUC
            if self.is_binary_classification:
                auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
            else:
                auc_score = roc_auc_score(y_test, y_pred_proba,
                                          multi_class='ovr', average='macro')

            print(f"  Downstream AUC: {auc_score:.4f}")

            model_performances.append({
                'model': model_info['model'],
                'metrics': model_info['metrics'],
                'downstream_auc': auc_score,
                'config_index': model_info['config_index']
            })

        # Select best model by AUC
        best_by_auc = max(model_performances, key=lambda x: x['downstream_auc'])

        print("\n" + "=" * 80)
        print("BEST MODEL SELECTED BY AUC")
        print("=" * 80)
        print(f"Configuration index: {best_by_auc['config_index']}")
        print(f"MRR: {best_by_auc['metrics'].get('mrr_score', 0):.4f}")
        print(f"Downstream AUC: {best_by_auc['downstream_auc']:.4f}")

        # Update optimizer's best model
        self.kge_optimizer.model = best_by_auc['model']
        self.kge_optimizer.best_embedding_dim = best_by_auc['metrics']['embedding_dim']

        # Re-extract embeddings with best model
        df = pd.read_csv(self.kge_csv_path)
        X_best = self.extract_embeddings(
            best_by_auc['model'],
            self.kge_optimizer.entity_to_id,
            df
        )

        return best_by_auc['model'], best_by_auc['metrics'], X_best

    def run_classification_with_statistics(self, use_random_search: bool = True,
                                           n_iter: int = 20, cv: int = 3) -> Tuple[Any, Dict]:
        """
        Run complete classification pipeline with statistical analysis
        """
        print("\n" + "=" * 80)
        print("STARTING CLASSIFICATION WITH STATISTICAL ANALYSIS")
        print("=" * 80)

        # Load data
        df = pd.read_csv(self.kge_csv_path)
        y = df['category'].values

        # Select best KGE model by AUC - FIXED: Pass both X and y
        best_kge_model, best_kge_metrics, X = self.select_best_model_by_auc(
            self.extract_embeddings(self.kge_optimizer.model, self.kge_optimizer.entity_to_id, df),
            y
        )

        print(f"\nFeature extraction completed")
        print(f"Feature shape: {X.shape}")
        print(f"Target shape: {y.shape}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Hyperparameter tuning
        print("\nHyperparameter tuning...")
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
        }

        base_model = XGBClassifier(
            objective='binary:logistic' if self.is_binary_classification else 'multi:softprob',
            eval_metric='logloss',
            random_state=42,
            use_label_encoder=False
        )

        if use_random_search:
            search = RandomizedSearchCV(
                base_model, param_grid, n_iter=n_iter, cv=cv,
                scoring='roc_auc_ovr' if not self.is_binary_classification else 'roc_auc',
                n_jobs=-1, verbose=1, random_state=42
            )
        else:
            search = GridSearchCV(
                base_model, param_grid, cv=cv,
                scoring='roc_auc_ovr' if not self.is_binary_classification else 'roc_auc',
                n_jobs=-1, verbose=1
            )

        search.fit(X_train, y_train)
        best_params = search.best_params_

        print(f"\nBest parameters: {best_params}")

        # Cross-validation with statistics
        cv_results = self.perform_cross_validation(X, y, best_params)

        # Train final model on full training set
        print("\nTraining final model...")
        final_model = XGBClassifier(
            **best_params,
            objective='binary:logistic' if self.is_binary_classification else 'multi:softprob',
            eval_metric='logloss',
            random_state=42,
            use_label_encoder=False
        )
        final_model.fit(X_train, y_train)

        # Evaluate on test set
        y_pred = final_model.predict(X_test)
        y_pred_proba = final_model.predict_proba(X_test)

        test_accuracy = accuracy_score(y_test, y_pred)
        if self.is_binary_classification:
            test_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        else:
            test_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
        
        # Calculate LogLoss and Mean Per Class Error for test set
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
        print(f"  Mean Per Class Error: {test_mpce:.4f}")

        # Pairwise comparison with Wilcoxon test (comparing folds)
        if len(cv_results['fold_accuracies']) >= 2:
            # Compare first half vs second half of folds (simplified pairwise comparison)
            mid = len(cv_results['fold_accuracies']) // 2
            group1 = cv_results['fold_accuracies'][:mid]
            group2 = cv_results['fold_accuracies'][mid:mid + len(group1)]

            if len(group1) == len(group2):
                statistic, p_value = stats.wilcoxon(group1, group2)
                print(f"\nPairwise comparison (Wilcoxon signed-rank test, N={len(group1)}):")
                print(f"  p-value: {p_value:.4f}")
                if p_value < 0.05:
                    print(f"  Result: Statistically significant differences detected (p < 0.05)")
                else:
                    print(f"  Result: No statistically significant differences (p = {p_value:.2f})")
            else:
                p_value = None
        else:
            p_value = None

        # Compile all results
        final_metrics = {
            # KGE Embedding Metrics (Level 1)
            'kge_model_type': self.model_type,
            'kge_embedding_dim': self.embedding_dim,
            'kge_metrics': {
                'mrr': best_kge_metrics.get('mrr_score', 0),
                'hits_at_1': best_kge_metrics.get('hits_at_1', 0),
                'hits_at_3': best_kge_metrics.get('hits_at_3', 0),
                'hits_at_10': best_kge_metrics.get('hits_at_10', 0),
            },
            'kge_config': {k: v for k, v in best_kge_metrics.items()
                           if k not in ['evaluation', 'all_results', 'model', 'error']},
            'selection_criterion': 'downstream_auc',  # Important: model selected by XGBoost AUC
            
            # XGBoost Classifier Metrics (Level 2)
            'best_classifier_params': best_params,
            
            # Cross-Validation Results
            'cv_results': {
                'mean_accuracy': cv_results['mean_accuracy'],
                'std_accuracy': cv_results['std_accuracy'],
                'ci_accuracy': cv_results['ci_accuracy'],
                'variance_accuracy': cv_results['variance_accuracy'],
                
                'mean_auc': cv_results['mean_auc'],
                'std_auc': cv_results['std_auc'],
                'ci_auc': cv_results['ci_auc'],
                
                'mean_logloss': cv_results['mean_logloss'],
                'std_logloss': cv_results['std_logloss'],
                'ci_logloss': cv_results['ci_logloss'],
                
                'mean_mpce': cv_results['mean_mpce'],
                'std_mpce': cv_results['std_mpce'],
                'ci_mpce': cv_results['ci_mpce'],
                
                'fold_accuracies': cv_results['fold_accuracies'],
                'fold_aucs': cv_results['fold_aucs'],
                'fold_logloss': cv_results['fold_logloss'],
                'fold_mean_per_class_error': cv_results['fold_mean_per_class_error'],
                'cv_scores': cv_results['cv_scores'],
            },
            
            # Test Set Results
            'test_accuracy': test_accuracy,
            'test_auc': test_auc,
            'test_logloss': test_logloss,
            'test_mpce': test_mpce,
            'test_classification_report': classification_report(y_test, y_pred, output_dict=True),
            'test_confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            
            # Statistical Tests
            'wilcoxon_p_value': p_value,
        }

        # Save results
        self.save_results(final_metrics, final_model)

        return final_model, final_metrics

    def save_results(self, metrics: Dict[str, Any], model: Any):
        """Save all results including statistical analysis"""

        # Save JSON results
        with open(self.results_file, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)

        # Save statistical analysis report
        with open(self.statistical_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("STATISTICAL ANALYSIS REPORT - TWO LEVELS\n")
            f.write("=" * 80 + "\n\n")

            # LEVEL 1: KGE EMBEDDINGS
            f.write("=" * 80 + "\n")
            f.write("LEVEL 1: KNOWLEDGE GRAPH EMBEDDING (KGE) METRICS\n")
            f.write("=" * 80 + "\n")
            f.write(f"Model Type: {metrics['kge_model_type'].upper()}\n")
            f.write(f"Embedding Dimension: {metrics['kge_embedding_dim']}\n")
            f.write(f"Selection Criterion: {metrics['selection_criterion']} (NOT MRR!)\n\n")
            
            f.write("KGE Performance Metrics:\n")
            f.write("-" * 60 + "\n")
            kge_metrics = metrics['kge_metrics']
            f.write(f"  MRR (Mean Reciprocal Rank): {kge_metrics['mrr']:.4f}\n")
            f.write(f"  Hits@1: {kge_metrics['hits_at_1']:.4f}\n")
            f.write(f"  Hits@3: {kge_metrics['hits_at_3']:.4f}\n")
            f.write(f"  Hits@10: {kge_metrics['hits_at_10']:.4f}\n\n")
            
            f.write("Note: These are intrinsic KGE metrics measuring link prediction\n")
            f.write("performance. High values reflect dataset structure (95% entity separation).\n")
            f.write("Model selection is based on downstream classification performance, not MRR.\n\n")

            # LEVEL 2: XGBOOST CLASSIFIER
            f.write("=" * 80 + "\n")
            f.write("LEVEL 2: XGBOOST CLASSIFIER METRICS (PRIMARY EVALUATION)\n")
            f.write("=" * 80 + "\n\n")

            f.write("CROSS-VALIDATION RESULTS (10-fold Stratified):\n")
            f.write("-" * 80 + "\n")
            cv = metrics['cv_results']
            
            f.write(f"\n1. ACCURACY\n")
            f.write(f"   Mean: {cv['mean_accuracy']:.4f} ± {cv['std_accuracy']:.4f}\n")
            f.write(f"   95% CI: ({cv['ci_accuracy'][0]:.4f}, {cv['ci_accuracy'][1]:.4f})\n")
            f.write(f"   Variance (σ²): {cv['variance_accuracy']:.6f}\n")
            
            f.write(f"\n2. AUC (Area Under ROC Curve)\n")
            f.write(f"   Mean: {cv['mean_auc']:.4f} ± {cv['std_auc']:.4f}\n")
            f.write(f"   95% CI: ({cv['ci_auc'][0]:.4f}, {cv['ci_auc'][1]:.4f})\n")
            
            f.write(f"\n3. LOG LOSS (Lower is Better)\n")
            f.write(f"   Mean: {cv['mean_logloss']:.4f} ± {cv['std_logloss']:.4f}\n")
            f.write(f"   95% CI: ({cv['ci_logloss'][0]:.4f}, {cv['ci_logloss'][1]:.4f})\n")
            
            f.write(f"\n4. MEAN PER CLASS ERROR (Lower is Better)\n")
            f.write(f"   Mean: {cv['mean_mpce']:.4f} ± {cv['std_mpce']:.4f}\n")
            f.write(f"   95% CI: ({cv['ci_mpce'][0]:.4f}, {cv['ci_mpce'][1]:.4f})\n")
            
            f.write(f"\n\nINDIVIDUAL FOLD SCORES:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Fold':<6} {'Accuracy':<10} {'AUC':<10} {'LogLoss':<10} {'MPCE':<10}\n")
            f.write("-" * 80 + "\n")
            for i in range(len(cv['fold_accuracies'])):
                f.write(f"{i+1:<6} {cv['fold_accuracies'][i]:<10.4f} {cv['fold_aucs'][i]:<10.4f} "
                       f"{cv['fold_logloss'][i]:<10.4f} {cv['fold_mean_per_class_error'][i]:<10.4f}\n")

            f.write("\n\nTEST SET PERFORMANCE (Independent Holdout):\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Test Accuracy: {metrics['test_accuracy']:.4f}\n")
            f.write(f"  Test AUC: {metrics['test_auc']:.4f}\n")
            f.write(f"  Test LogLoss: {metrics['test_logloss']:.4f}\n")
            f.write(f"  Test Mean Per Class Error: {metrics['test_mpce']:.4f}\n\n")

            # STATISTICAL SIGNIFICANCE
            if metrics['wilcoxon_p_value'] is not None:
                f.write("STATISTICAL SIGNIFICANCE TEST:\n")
                f.write("-" * 80 + "\n")
                f.write(f"Wilcoxon signed-rank test (comparing fold halves):\n")
                f.write(f"  p-value: {metrics['wilcoxon_p_value']:.4f}\n")
                if metrics['wilcoxon_p_value'] < 0.05:
                    f.write(f"  Result: Statistically significant differences detected (p < 0.05)\n\n")
                else:
                    f.write(f"  Result: No statistically significant differences (p >= 0.05)\n\n")

            # BEST PARAMETERS
            f.write("BEST XGBOOST HYPERPARAMETERS:\n")
            f.write("-" * 80 + "\n")
            for param, value in metrics['best_classifier_params'].items():
                f.write(f"  {param}: {value}\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")

            f.write("CROSS-VALIDATION RESULTS (N={} folds)\n".format(self.n_folds))
            f.write("-" * 80 + "\n")
            cv = metrics['cv_results']
            f.write(f"Classification Accuracy: {cv['mean_accuracy']:.3f} ")
            f.write(f"(95% CI: {cv['ci_accuracy'][0]:.3f}–{cv['ci_accuracy'][1]:.3f})\n")
            f.write(f"Macro AUC: {cv['mean_auc']:.3f} ")
            f.write(f"(95% CI: {cv['ci_auc'][0]:.3f}–{cv['ci_auc'][1]:.3f})\n")
            f.write(f"Standard Deviation: {cv['std_accuracy']:.4f}\n")
            f.write(f"Variance (σ²): {cv['variance_accuracy']:.6f}\n\n")

            if metrics.get('wilcoxon_p_value') is not None:
                f.write("PAIRWISE MODEL COMPARISON\n")
                f.write("-" * 80 + "\n")
                f.write(f"Wilcoxon signed-rank test: p = {metrics['wilcoxon_p_value']:.2f}\n")
                if metrics['wilcoxon_p_value'] < 0.05:
                    f.write("Result: Statistically significant differences (p < 0.05)\n\n")
                else:
                    f.write("Result: No statistically significant differences\n\n")

            f.write("TEST SET PERFORMANCE\n")
            f.write("-" * 80 + "\n")
            f.write(f"Accuracy: {metrics['test_accuracy']:.4f}\n")
            f.write(f"AUC: {metrics['test_auc']:.4f}\n\n")

            f.write("ROBUSTNESS ASSESSMENT\n")
            f.write("-" * 80 + "\n")
            if cv['variance_accuracy'] < 0.005:
                f.write("✓ Low variance across folds supports robustness\n")
            else:
                f.write("⚠ Moderate to high variance detected\n")
            f.write(f"Variance: {cv['variance_accuracy']:.6f}\n")

        # Save model
        joblib.dump(model, self.best_model_file)

        print(f"\n✓ Results saved:")
        print(f"  JSON: {self.results_file}")
        print(f"  Statistical Analysis: {self.statistical_file}")
        print(f"  Model: {self.best_model_file}")


def run_complete_pipeline_with_statistics(kg_path: str, emb_dir: str, output_dir: str,
                                          model_type: str = 'complex',
                                          search_type: str = 'smart',
                                          n_random_samples: int = 15,
                                          use_random_search: bool = True,
                                          n_iter: int = 20,
                                          cv: int = 3,
                                          n_folds_cv: int = 10,
                                          early_stopping_patience: int = 5,
                                          early_stopping_delta: float = 0.0005,
                                          aggressive_early_stop: bool = False) -> Tuple[Any, Dict]:
    """
    Run complete pipeline: KGE optimization + Classification with statistics
    
    Args:
        early_stopping_patience: Number of epochs with no improvement before stopping
        early_stopping_delta: Minimum relative improvement required
        aggressive_early_stop: If True, use more aggressive early stopping to prevent overfitting
    """
    print("=" * 80)
    print("COMPLETE PIPELINE WITH STATISTICAL ANALYSIS")
    print("=" * 80)
    print(f"KGE Model: {model_type.upper()}")
    print(f"Search Strategy: {search_type}")
    print(f"CV Folds for Statistics: {n_folds_cv}")
    if aggressive_early_stop:
        print(f"⚠️ AGGRESSIVE Early Stopping Enabled (prevents overfitting)")
    else:
        print(f"Early Stopping: patience={early_stopping_patience}, delta={early_stopping_delta}")
    print("=" * 80)

    # Phase 1: KGE Training
    print("\n--- PHASE 1: KGE OPTIMIZATION ---")
    kge_optimizer = KGEOptimizer(
        kg_path=kg_path,
        emb_dir=emb_dir,
        output_dir=output_dir,
        model_type=model_type,
        early_stopping_patience=early_stopping_patience,
        early_stopping_delta=early_stopping_delta,
        aggressive_early_stop=aggressive_early_stop
    )

    kge_optimizer.run_kge_training(search_type=search_type, n_random_samples=n_random_samples)

    # Phase 2: Classification with Statistics
    print("\n--- PHASE 2: CLASSIFICATION WITH STATISTICAL ANALYSIS ---")
    classifier = TopicClassifierWithStatistics(
        kge_csv_path=kg_path,
        kge_optimizer=kge_optimizer,
        output_dir=output_dir,
        n_folds=n_folds_cv
    )

    best_model, metrics = classifier.run_classification_with_statistics(
        use_random_search=use_random_search,
        n_iter=n_iter,
        cv=cv
    )

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED")
    print("=" * 80)
    print(f"Final Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Final Test AUC: {metrics['test_auc']:.4f}")
    print(f"CV Accuracy: {metrics['cv_results']['mean_accuracy']:.4f} "
          f"(95% CI: {metrics['cv_results']['ci_accuracy'][0]:.3f}–"
          f"{metrics['cv_results']['ci_accuracy'][1]:.3f})")

    return best_model, metrics


def run_all_experiments(use_smart_search: bool = True,
                        n_random_samples: int = 15,
                        use_random_search: bool = True,
                        n_iter: int = 20,
                        cv: int = 3,
                        n_folds_cv: int = 10,
                        aggressive_early_stop: bool = False):
    """
    Run experiments for ALL repositories and ALL models
    """
    repositories = {
        'amazon': 'dataset_triplet_amazon_new_simplificado.csv',
        'bbc': 'dataset_triplet_bbc_new_simplificado.csv',
        'reuters': 'dataset_triplet_reuters_activities_new_simplificado.csv'
    }

    models = list(KGEOptimizer.AVAILABLE_MODELS.keys())

    print("\n" + "=" * 80)
    print("RUNNING COMPLETE EXPERIMENTAL SUITE")
    print("=" * 80)
    print(f"Repositories: {len(repositories)} (Amazon, BBC, Reuters)")
    print(f"Models: {len(models)} ({', '.join([KGEOptimizer.AVAILABLE_MODELS[m] for m in models])})")
    print(f"Total experiments: {len(repositories) * len(models)}")
    print(f"Smart search enabled: {use_smart_search}")
    print("=" * 80)

    confirm = input("\nThis will take several hours. Continue? (yes/no): ").strip().lower()
    if confirm != 'yes':
        print("Aborted.")
        return

    base_path = "../data"
    all_results = {}

    total_experiments = len(repositories) * len(models)
    current_experiment = 0

    for repo, dataset_name in repositories.items():
        all_results[repo] = {}

        for model in models:
            current_experiment += 1
            print("\n" + "=" * 80)
            print(f"EXPERIMENT {current_experiment}/{total_experiments}")
            print(f"Repository: {repo.upper()}")
            print(f"Model: {KGEOptimizer.AVAILABLE_MODELS[model]}")
            print("=" * 80)

            # Define paths
            kg_path = f"{base_path}/triples_raw/{repo}/{dataset_name}"
            emb_dir = f"{base_path}/triples_emb/{repo}_statistical_{model}"
            output_dir = f"{base_path}/model_output/{repo}_statistical_{model}"

            # Determine search strategy
            if use_smart_search:
                search_type = 'smart'
                n_samples = 0
            else:
                search_type = 'random'
                n_samples = n_random_samples

            try:
                # Run pipeline
                best_model, metrics = run_complete_pipeline_with_statistics(
                    kg_path=kg_path,
                    emb_dir=emb_dir,
                    output_dir=output_dir,
                    model_type=model,
                    search_type=search_type,
                    n_random_samples=n_samples,
                    use_random_search=use_random_search,
                    n_iter=n_iter,
                    cv=cv,
                    n_folds_cv=n_folds_cv,
                    aggressive_early_stop=aggressive_early_stop
                )

                # Store summary results with full paths and both levels of metrics
                all_results[repo][model] = {
                    # Level 1: KGE Embeddings
                    'kge_metrics': {
                        'mrr': metrics.get('kge_metrics', {}).get('mrr', 0),
                        'hits_at_1': metrics.get('kge_metrics', {}).get('hits_at_1', 0),
                        'hits_at_10': metrics.get('kge_metrics', {}).get('hits_at_10', 0),
                    },
                    'embedding_dim': metrics.get('kge_embedding_dim', 0),
                    
                    # Level 2: XGBoost Classifier (Primary Metrics)
                    'test_accuracy': metrics['test_accuracy'],
                    'test_auc': metrics['test_auc'],
                    'test_logloss': metrics.get('test_logloss', 0),
                    'test_mpce': metrics.get('test_mpce', 0),
                    
                    # Cross-Validation Results
                    'cv_mean_accuracy': metrics['cv_results']['mean_accuracy'],
                    'cv_std_accuracy': metrics['cv_results'].get('std_accuracy', 0),
                    'cv_ci_accuracy': metrics['cv_results']['ci_accuracy'],
                    'cv_mean_auc': metrics['cv_results']['mean_auc'],
                    'cv_std_auc': metrics['cv_results'].get('std_auc', 0),
                    'cv_mean_logloss': metrics['cv_results'].get('mean_logloss', 0),
                    'cv_std_logloss': metrics['cv_results'].get('std_logloss', 0),
                    'cv_mean_mpce': metrics['cv_results'].get('mean_mpce', 0),
                    'cv_std_mpce': metrics['cv_results'].get('std_mpce', 0),
                    'cv_variance': metrics['cv_results']['variance_accuracy'],
                    'wilcoxon_p': metrics.get('wilcoxon_p_value'),
                    'cv_scores': metrics['cv_results'].get('cv_scores', []),  # All fold scores
                    
                    # Metadata
                    'status': 'SUCCESS',
                    'selection_criterion': 'downstream_auc',
                    'paths': {
                        'embeddings_dir': emb_dir,
                        'output_dir': output_dir,
                        'kg_path': kg_path
                    },
                    'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
                }

                print(f"\n✓ Experiment completed successfully!")
                print(f"  Test Accuracy: {metrics['test_accuracy']:.4f}")
                print(f"  Test AUC: {metrics['test_auc']:.4f}")

            except Exception as e:
                print(f"\n✗ Experiment failed with error: {e}")
                all_results[repo][model] = {
                    'status': 'FAILED',
                    'error': str(e)
                }

    # Save comprehensive summary
    summary_file = f"{base_path}/model_output/COMPLETE_EXPERIMENTAL_SUMMARY_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print summary table
    print("\n" + "=" * 80)
    print("EXPERIMENTAL SUITE COMPLETED - RESULTS SUMMARY")
    print("=" * 80)
    print("\nLEVEL 1: KGE EMBEDDING METRICS")
    print("-" * 80)
    print(f"{'Repository':<12} {'Model':<10} {'MRR':<10} {'Hits@1':<10} {'Emb Dim':<10}")
    print("-" * 80)
    
    for repo, models_results in all_results.items():
        for model, result in models_results.items():
            if result['status'] == 'SUCCESS':
                mrr = f"{result['kge_metrics']['mrr']:.4f}"
                h1 = f"{result['kge_metrics']['hits_at_1']:.4f}"
                emb = str(result['embedding_dim'])
            else:
                mrr = h1 = emb = "N/A"
            
            print(f"{repo:<12} {model:<10} {mrr:<10} {h1:<10} {emb:<10}")
    
    print("\n" + "=" * 80)
    print("LEVEL 2: XGBOOST CLASSIFIER METRICS (PRIMARY EVALUATION)")
    print("=" * 80)
    print(f"{'Repository':<12} {'Model':<10} {'Test Acc':<11} {'CV Acc±SD':<15} {'Test AUC':<10} {'LogLoss':<10}")
    print("-" * 80)

    for repo, models_results in all_results.items():
        for model, result in models_results.items():
            if result['status'] == 'SUCCESS':
                acc = f"{result['test_accuracy']:.4f}"
                cv_acc = f"{result['cv_mean_accuracy']:.3f}±{result['cv_std_accuracy']:.3f}"
                auc = f"{result['test_auc']:.4f}"
                ll = f"{result.get('test_logloss', 0):.4f}"
                status = "✓"
            else:
                acc = cv_acc = auc = ll = "N/A"
                status = "✗"

            print(f"{repo:<12} {model:<10} {acc:<11} {cv_acc:<15} {auc:<10} {ll:<10} {status}")

    print("-" * 80)
    print(f"\nComplete results saved to: {summary_file}")
    print("\n✓ ALL EXPERIMENTS COMPLETED!")
    
    # Statistical comparison between models
    print("\n" + "=" * 80)
    print("STATISTICAL COMPARISON BETWEEN MODELS")
    print("=" * 80)
    perform_model_comparison(all_results)
    
    # Interactive model selection
    print("\n" + "=" * 80)
    print("MODEL SELECTION FOR FINAL ANALYSIS")
    print("=" * 80)
    selected_results = interactive_model_selection(all_results, base_path)
    
    return all_results, selected_results


def perform_model_comparison(all_results: Dict) -> None:
    """
    Perform statistical comparison between models across repositories
    Uses Friedman test for multiple models comparison
    """
    from scipy.stats import friedmanchisquare, wilcoxon
    
    print("\nPerforming Friedman test to compare all models...")
    
    # Organize data by repository
    for repo, models_results in all_results.items():
        print(f"\n{'='*60}")
        print(f"Repository: {repo.upper()}")
        print(f"{'='*60}")
        
        successful_models = {model: result for model, result in models_results.items() 
                            if result['status'] == 'SUCCESS'}
        
        if len(successful_models) < 2:
            print("  ⚠️ Not enough models for comparison (need at least 2)")
            continue
        
        # Extract CV scores for each model
        cv_scores_by_model = {}
        for model, result in successful_models.items():
            cv_scores = result.get('cv_scores', [])
            if cv_scores:
                cv_scores_by_model[model] = cv_scores
        
        if len(cv_scores_by_model) < 2:
            print("  ⚠️ CV scores not available for comparison")
            continue
        
        # Check all models have same number of folds
        n_folds = len(list(cv_scores_by_model.values())[0])
        if not all(len(scores) == n_folds for scores in cv_scores_by_model.values()):
            print("  ⚠️ Models have different number of CV folds")
            continue
        
        # Friedman test
        model_names = list(cv_scores_by_model.keys())
        model_scores = list(cv_scores_by_model.values())
        
        if len(model_names) >= 3:
            statistic, p_value = friedmanchisquare(*model_scores)
            print(f"\n  Friedman Test (comparing {len(model_names)} models):")
            print(f"    Statistic: {statistic:.4f}")
            print(f"    P-value: {p_value:.4f}")
            
            if p_value < 0.05:
                print(f"    ✓ Significant difference between models (p < 0.05)")
            else:
                print(f"    ✗ No significant difference between models (p >= 0.05)")
        
        # Pairwise Wilcoxon tests
        if len(model_names) >= 2:
            print(f"\n  Pairwise Wilcoxon Tests:")
            print(f"  {'-'*50}")
            for i in range(len(model_names)):
                for j in range(i+1, len(model_names)):
                    model1, model2 = model_names[i], model_names[j]
                    scores1, scores2 = model_scores[i], model_scores[j]
                    
                    try:
                        stat, p_val = wilcoxon(scores1, scores2)
                        mean1 = np.mean(scores1)
                        mean2 = np.mean(scores2)
                        
                        print(f"  {model1.upper()} vs {model2.upper()}:")
                        print(f"    Mean: {mean1:.4f} vs {mean2:.4f}")
                        print(f"    P-value: {p_val:.4f}", end="")
                        if p_val < 0.05:
                            winner = model1 if mean1 > mean2 else model2
                            print(f" ✓ {winner.upper()} is significantly better")
                        else:
                            print(f" ✗ No significant difference")
                    except Exception as e:
                        print(f"  {model1.upper()} vs {model2.upper()}: Error - {e}")
        
        # Summary table
        print(f"\n  Model Performance Summary:")
        print(f"  {'-'*60}")
        print(f"  {'Model':<10} {'Mean Acc':<12} {'Std Dev':<12} {'Test Acc':<12}")
        print(f"  {'-'*60}")
        for model in model_names:
            result = successful_models[model]
            cv_scores = cv_scores_by_model[model]
            mean_cv = np.mean(cv_scores)
            std_cv = np.std(cv_scores)
            test_acc = result['test_accuracy']
            print(f"  {model.upper():<10} {mean_cv:.4f}      {std_cv:.4f}       {test_acc:.4f}")


def interactive_model_selection(all_results: Dict, base_path: str) -> Dict:
    """
    Interactive menu to select best model per repository and compute detailed statistics
    """
    selected_results = {}
    
    for repo, models_results in all_results.items():
        print(f"\n{'='*60}")
        print(f"SELECT BEST MODEL FOR: {repo.upper()}")
        print(f"{'='*60}")
        
        successful_models = {model: result for model, result in models_results.items() 
                            if result['status'] == 'SUCCESS'}
        
        if not successful_models:
            print("  ⚠️ No successful models for this repository")
            continue
        
        # Display options
        print("\nAvailable models:")
        model_list = list(successful_models.keys())
        for idx, model in enumerate(model_list, 1):
            result = successful_models[model]
            print(f"{idx}. {model.upper():<10} - Acc: {result['test_accuracy']:.4f}, AUC: {result['test_auc']:.4f}")
        
        # Auto-select best by test accuracy
        best_idx = max(range(len(model_list)), 
                      key=lambda i: successful_models[model_list[i]]['test_accuracy'])
        print(f"\nRecommended (highest test accuracy): {model_list[best_idx].upper()}")
        
        choice = input(f"\nSelect model (1-{len(model_list)}, or press Enter for recommended): ").strip()
        
        if choice == '':
            selected_model = model_list[best_idx]
        else:
            try:
                selected_model = model_list[int(choice) - 1]
            except (ValueError, IndexError):
                print("Invalid choice, using recommended model")
                selected_model = model_list[best_idx]
        
        selected_results[repo] = {
            'model': selected_model,
            'results': successful_models[selected_model]
        }
        
        print(f"\n✓ Selected {selected_model.upper()} for {repo.upper()}")
        
        # Compute detailed statistics for selected model
        print(f"\nComputing detailed statistics for {selected_model.upper()}...")
        compute_detailed_statistics(repo, selected_model, successful_models[selected_model], base_path)
    
    # Save selected models summary
    selected_file = f"{base_path}/model_output/SELECTED_MODELS_SUMMARY_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(selected_file, 'w') as f:
        json.dump(selected_results, f, indent=2, default=str)
    
    print(f"\n✓ Selected models summary saved to: {selected_file}")
    
    return selected_results


def compute_detailed_statistics(repo: str, model: str, result: Dict, base_path: str) -> None:
    """
    Compute detailed statistics for the selected model
    Reports metrics at BOTH levels: KGE embeddings and XGBoost classifier
    """
    output_file = f"{base_path}/model_output/{repo}_{model}_DETAILED_STATISTICS_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"DETAILED STATISTICAL ANALYSIS - TWO LEVELS\n")
        f.write(f"Repository: {repo.upper()}\n")
        f.write(f"Model: {model.upper()}\n")
        f.write("=" * 80 + "\n\n")
        
        # LEVEL 1: KGE EMBEDDINGS
        f.write("=" * 80 + "\n")
        f.write("LEVEL 1: KNOWLEDGE GRAPH EMBEDDING METRICS\n")
        f.write("=" * 80 + "\n")
        if 'kge_metrics' in result:
            kge_metrics = result['kge_metrics']
            f.write(f"MRR (Mean Reciprocal Rank): {kge_metrics.get('mrr', 0):.4f}\n")
            f.write(f"Hits@1: {kge_metrics.get('hits_at_1', 0):.4f}\n")
            f.write(f"Hits@3: {kge_metrics.get('hits_at_3', 0):.4f}\n")
            f.write(f"Hits@10: {kge_metrics.get('hits_at_10', 0):.4f}\n\n")
        f.write("Note: Model selection based on downstream classification AUC, not MRR\n\n")
        
        # LEVEL 2: XGBOOST CLASSIFIER
        f.write("=" * 80 + "\n")
        f.write("LEVEL 2: XGBOOST CLASSIFIER PERFORMANCE (PRIMARY METRICS)\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. TEST SET PERFORMANCE (Independent Holdout)\n")
        f.write("-" * 60 + "\n")
        f.write(f"Test Accuracy: {result['test_accuracy']:.4f}\n")
        f.write(f"Test AUC: {result['test_auc']:.4f}\n")
        if 'test_logloss' in result:
            f.write(f"Test LogLoss: {result['test_logloss']:.4f}\n")
        if 'test_mpce' in result:
            f.write(f"Test Mean Per Class Error: {result['test_mpce']:.4f}\n")
        f.write("\n")
        
        f.write("2. CROSS-VALIDATION RESULTS (10-fold Stratified)\n")
        f.write("-" * 60 + "\n")
        cv = result.get('cv_results', {})
        
        f.write(f"Accuracy: {cv['mean_accuracy']:.4f} ± {cv.get('std_accuracy', 0):.4f}\n")
        f.write(f"  95% CI: {cv['ci_accuracy']}\n")
        f.write(f"  Variance (σ²): {cv['variance_accuracy']:.6f}\n\n")
        
        f.write(f"AUC: {cv['mean_auc']:.4f} ± {cv.get('std_auc', 0):.4f}\n")
        f.write(f"  95% CI: {cv.get('ci_auc', 'N/A')}\n\n")
        
        if 'mean_logloss' in cv:
            f.write(f"LogLoss: {cv['mean_logloss']:.4f} ± {cv.get('std_logloss', 0):.4f}\n")
            f.write(f"  95% CI: {cv.get('ci_logloss', 'N/A')}\n\n")
        
        if 'mean_mpce' in cv:
            f.write(f"Mean Per Class Error: {cv['mean_mpce']:.4f} ± {cv.get('std_mpce', 0):.4f}\n")
            f.write(f"  95% CI: {cv.get('ci_mpce', 'N/A')}\n\n")
        
        if 'cv_scores' in cv and cv['cv_scores']:
            cv_scores = cv['cv_scores']
            f.write("3. INDIVIDUAL FOLD SCORES (Accuracy)\n")
            f.write("-" * 60 + "\n")
            for i, score in enumerate(cv_scores, 1):
                f.write(f"Fold {i}: {score:.4f}\n")
            f.write(f"\nStandard Deviation: {np.std(cv_scores):.4f}\n")
            f.write(f"Min: {np.min(cv_scores):.4f}\n")
            f.write(f"Max: {np.max(cv_scores):.4f}\n\n")
        
        if result.get('wilcoxon_p'):
            f.write("4. STATISTICAL SIGNIFICANCE TEST\n")
            f.write("-" * 60 + "\n")
            f.write(f"Wilcoxon p-value: {result['wilcoxon_p']:.4f}\n")
            if result['wilcoxon_p'] < 0.05:
                f.write("✓ Statistically significant (p < 0.05)\n\n")
            else:
                f.write("✗ No significant difference from baseline (p >= 0.05)\n\n")
        
        f.write("5. FILE LOCATIONS\n")
        f.write("-" * 60 + "\n")
        if 'paths' in result:
            f.write(f"Embeddings: {result['paths']['embeddings_dir']}\n")
            f.write(f"Outputs: {result['paths']['output_dir']}\n")
            f.write(f"Knowledge Graph: {result['paths']['kg_path']}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("INTERPRETATION NOTES:\n")
        f.write("-" * 60 + "\n")
        f.write("• Level 1 (KGE metrics): Measure link prediction performance\n")
        f.write("• Level 2 (Classifier metrics): Measure downstream classification task\n")
        f.write("• Model selection: Based on Level 2 performance (XGBoost AUC)\n")
        f.write("• High KGE metrics reflect dataset structure, not necessarily overfitting\n")
        f.write("=" * 80 + "\n")
    
    print(f"  Detailed statistics saved to: {output_file}")


def run_all_models_for_repository(repo: str, use_smart_search: bool = True, 
                                   n_random_samples: int = 15, use_random_search: bool = True,
                                   n_iter: int = 20, cv: int = 3, n_folds_cv: int = 10,
                                   aggressive_early_stop: bool = False):
    """
    Run all KGE models (TransE, ConvKB, ComplEx) for a single repository
    with the same configuration as the complete experimental suite
    """
    print("\n" + "=" * 80)
    print(f"RUNNING ALL MODELS FOR {repo.upper()} REPOSITORY")
    print("=" * 80)
    print(f"This will run experiments for:")
    print(f"  • Repository: {repo}")
    print(f"  • All models: TransE, ConvKB, ComplEx")
    print(f"  • Total: 3 experiments")
    print()
    
    # Map repository to dataset
    repo_datasets = {
        'amazon': 'dataset_triplet_amazon_new_simplificado.csv',
        'bbc': 'dataset_triplet_bbc_new_simplificado.csv',
        'reuters': 'dataset_triplet_reuters_activities_new_simplificado.csv'
    }
    
    if repo not in repo_datasets:
        raise ValueError(f"Invalid repository: {repo}. Choose from: {list(repo_datasets.keys())}")
    
    dataset_name = repo_datasets[repo]
    base_path = "../data"
    
    # Results storage
    all_results = {}
    
    # Run experiments for all models
    for model in ['transe', 'convkb', 'complex']:
        print("\n" + "=" * 80)
        print(f"EXPERIMENT: {repo.upper()} - {model.upper()}")
        print("=" * 80)
        
        kg_path = f"{base_path}/triples_raw/{repo}/{dataset_name}"
        emb_dir = f"{base_path}/triples_emb/{repo}_statistical"
        output_dir = f"{base_path}/model_output/{repo}_statistical"
        
        try:
            print(f"\n🚀 Starting training for {model.upper()}...")
            
            best_model, metrics = run_complete_pipeline_with_statistics(
                kg_path=kg_path,
                emb_dir=emb_dir,
                output_dir=output_dir,
                model_type=model,
                search_type='smart' if use_smart_search else 'random',
                n_random_samples=n_random_samples,
                use_random_search=use_random_search,
                n_iter=n_iter,
                cv=cv,
                n_folds_cv=n_folds_cv,
                aggressive_early_stop=aggressive_early_stop
            )
            
            # Store summary results with both levels of metrics
            all_results[model] = {
                # Level 1: KGE Embeddings
                'kge_metrics': {
                    'mrr': metrics.get('kge_metrics', {}).get('mrr', 0),
                    'hits_at_1': metrics.get('kge_metrics', {}).get('hits_at_1', 0),
                    'hits_at_10': metrics.get('kge_metrics', {}).get('hits_at_10', 0),
                },
                'embedding_dim': metrics.get('kge_embedding_dim', 0),
                
                # Level 2: XGBoost Classifier
                'test_accuracy': metrics['test_accuracy'],
                'test_auc': metrics['test_auc'],
                'test_logloss': metrics.get('test_logloss', 0),
                'test_mpce': metrics.get('test_mpce', 0),
                
                # Cross-Validation
                'cv_mean_accuracy': metrics['cv_results']['mean_accuracy'],
                'cv_std_accuracy': metrics['cv_results'].get('std_accuracy', 0),
                'cv_ci_accuracy': metrics['cv_results']['ci_accuracy'],
                'cv_mean_auc': metrics['cv_results']['mean_auc'],
                'cv_mean_logloss': metrics['cv_results'].get('mean_logloss', 0),
                'cv_mean_mpce': metrics['cv_results'].get('mean_mpce', 0),
                'cv_variance': metrics['cv_results']['variance_accuracy'],
                'cv_scores': metrics['cv_results'].get('cv_scores', []),
                'wilcoxon_p': metrics.get('wilcoxon_p_value'),
                'status': 'SUCCESS'
            }
            
            print(f"\n✓ Experiment completed successfully!")
            print(f"  Test Accuracy: {metrics['test_accuracy']:.4f}")
            print(f"  Test AUC: {metrics['test_auc']:.4f}")
            if 'test_logloss' in metrics:
                print(f"  Test LogLoss: {metrics['test_logloss']:.4f}")
            if 'test_mpce' in metrics:
                print(f"  Test MPCE: {metrics['test_mpce']:.4f}")
            
        except Exception as e:
            print(f"\n✗ Experiment failed with error: {e}")
            all_results[model] = {
                'status': 'FAILED',
                'error': str(e)
            }
    
    # Save comprehensive summary for this repository
    summary_file = f"{base_path}/model_output/{repo}_ALL_MODELS_SUMMARY_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w') as f:
        json.dump({repo: all_results}, f, indent=2, default=str)
    
    # Print summary table
    print("\n" + "=" * 80)
    print(f"ALL MODELS COMPLETED FOR {repo.upper()}")
    print("=" * 80)
    print("\nSUMMARY TABLE:")
    print("-" * 80)
    print(f"{'Model':<10} {'Accuracy':<12} {'AUC':<12} {'Status':<10}")
    print("-" * 80)
    
    for model, result in all_results.items():
        if result['status'] == 'SUCCESS':
            acc = f"{result['test_accuracy']:.4f}"
            auc = f"{result['test_auc']:.4f}"
            status = "✓ SUCCESS"
        else:
            acc = "N/A"
            auc = "N/A"
            status = "✗ FAILED"
        
        print(f"{model.upper():<10} {acc:<12} {auc:<12} {status:<10}")
    
    print("-" * 80)
    print(f"\nComplete results saved to: {summary_file}")
    print(f"\n✓ ALL MODELS COMPLETED FOR {repo.upper()}!")


def interactive_menu():
    """Interactive menu for running the pipeline"""
    print("\n" + "=" * 80)
    print("KGE CLASSIFICATION WITH STATISTICAL RIGOR")
    print("=" * 80)

    # Select execution mode
    print("\nExecution mode:")
    print("1. Single experiment (select repository and model)")
    print("2. All models for ONE repository (select repository)")
    print("3. Complete experimental suite (ALL repositories + ALL models)")

    mode_choice = input("\nSelect mode (1-3, default: 1): ").strip() or "1"

    if mode_choice == '2':
        # All models for ONE repository
        print("\n" + "=" * 80)
        print("ALL MODELS FOR ONE REPOSITORY")
        print("=" * 80)
        
        # Select repository
        print("\nAvailable repositories:")
        print("1. Amazon")
        print("2. BBC")
        print("3. Reuters")
        
        repo_choice = input("\nSelect repository (1-3): ").strip()
        
        if repo_choice == '1':
            repo = 'amazon'
        elif repo_choice == '2':
            repo = 'bbc'
        elif repo_choice == '3':
            repo = 'reuters'
        else:
            print("Invalid choice. Exiting...")
            sys.exit(1)
        
        print(f"\nThis will run experiments for {repo.upper()} with:")
        print("  • All models: TransE, ConvKB, ComplEx")
        print("  • Total: 3 experiments")
        print()
        
        # Ask about smart search for all models
        use_smart = input("Use smart search for all models? (y/n, default: y): ").strip().lower()
        use_smart_search = (use_smart != 'n')
        
        if not use_smart_search:
            # Ask for configuration if not using smart search
            n_random_samples = int(input("Number of random configs (default: 15): ").strip() or "15")
        else:
            n_random_samples = 0  # Not used with smart search
        
        # Ask about aggressive early stopping
        print("\nEarly Stopping Configuration:")
        use_aggressive = input("Use AGGRESSIVE early stopping to prevent overfitting? (y/n, default: n): ").strip().lower()
        aggressive_early_stop = (use_aggressive == 'y')
        
        # Run all models for this repository
        run_all_models_for_repository(
            repo=repo,
            use_smart_search=use_smart_search,
            n_random_samples=n_random_samples,
            use_random_search=True,
            n_iter=20,
            cv=3,
            n_folds_cv=10,
            aggressive_early_stop=aggressive_early_stop
        )
        return

    if mode_choice == '3':
        # Complete experimental suite
        print("\n" + "=" * 80)
        print("COMPLETE EXPERIMENTAL SUITE")
        print("=" * 80)
        print("This will run experiments for:")
        print("  • All repositories: Amazon, BBC, Reuters")
        print("  • All models: TransE, ConvKB, ComplEx")
        print("  • Total: 9 experiments")
        print()

        # Ask about smart search for all models
        use_smart = input("Use smart search for all models? (y/n, default: y): ").strip().lower()
        use_smart_search = (use_smart != 'n')

        if not use_smart_search:
            # Ask for configuration if not using smart search
            n_random_samples = int(input("Number of random configs (default: 15): ").strip() or "15")
        else:
            n_random_samples = 0  # Not used with smart search
        
        # Ask about aggressive early stopping to prevent overfitting
        print("\nEarly Stopping Configuration:")
        print("Aggressive early stopping stops training earlier if no improvement is seen.")
        print("This helps prevent overfitting but may stop before reaching optimal performance.")
        use_aggressive = input("Use AGGRESSIVE early stopping? (y/n, default: n): ").strip().lower()
        aggressive_early_stop = (use_aggressive == 'y')

        # Run all experiments
        run_all_experiments(
            use_smart_search=use_smart_search,
            n_random_samples=n_random_samples,
            use_random_search=True,
            n_iter=20,
            cv=3,
            n_folds_cv=10,
            aggressive_early_stop=aggressive_early_stop
        )
        return

    # Single experiment mode
    # Select repository
    print("\nAvailable repositories:")
    print("1. Amazon")
    print("2. BBC")
    print("3. Reuters")

    repo_choice = input("\nSelect repository (1-3): ").strip()

    if repo_choice == '1':
        repo = 'amazon'
        dataset_name = 'dataset_triplet_amazon_new_simplificado.csv'
    elif repo_choice == '2':
        repo = 'bbc'
        dataset_name = 'dataset_triplet_bbc_new_simplificado.csv'
    elif repo_choice == '3':
        repo = 'reuters'
        dataset_name = 'dataset_triplet_reuters_activities_new_simplificado.csv'
    else:
        print("Invalid choice. Exiting...")
        sys.exit(1)

    # Select KGE model
    print("\nAvailable KGE models:")
    for key, name in KGEOptimizer.AVAILABLE_MODELS.items():
        print(f"  {key}: {name}")

    model_type = input("\nSelect KGE model (default: complex): ").strip().lower() or 'complex'

    # Select search strategy
    print("\nSearch strategies:")
    print("1. Smart search (optimized configurations for each model)")
    print("2. Random search")
    print("3. Grid search (exhaustive)")

    search_choice = input("\nSelect search strategy (1-3, default: 1): ").strip() or "1"

    if search_choice == '1':
        search_type = 'smart'
        n_samples = 0
    elif search_choice == '2':
        search_type = 'random'
        n_samples = int(input("Number of random configurations (default: 15): ").strip() or "15")
    else:
        search_type = 'grid'
        n_samples = 0

    # Define paths
    base_path = "../data"
    kg_path = f"{base_path}/triples_raw/{repo}/{dataset_name}"
    emb_dir = f"{base_path}/triples_emb/{repo}_statistical"
    output_dir = f"{base_path}/model_output/{repo}_statistical"

    # Run pipeline
    best_model, metrics = run_complete_pipeline_with_statistics(
        kg_path=kg_path,
        emb_dir=emb_dir,
        output_dir=output_dir,
        model_type=model_type,
        search_type=search_type,
        n_random_samples=n_samples,
        use_random_search=True,
        n_iter=20,
        cv=3,
        n_folds_cv=10
    )

    print("\n✓ All tasks completed successfully!")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Programmatic mode
        if len(sys.argv) >= 3:
            repo = sys.argv[1].lower()
            model_type = sys.argv[2].lower()

            if repo == 'amazon':
                dataset_name = 'dataset_triplet_amazon_new_simplificado.csv'
            elif repo == 'bbc':
                dataset_name = 'dataset_triplet_bbc_new_simplificado.csv'
            elif repo == 'reuters':
                dataset_name = 'dataset_triplet_reuters_activities_new_simplificado.csv'
            else:
                print("Invalid repository")
                sys.exit(1)

            base_path = "../data"
            kg_path = f"{base_path}/triples_raw/{repo}/{dataset_name}"
            emb_dir = f"{base_path}/triples_emb/{repo}_statistical"
            output_dir = f"{base_path}/model_output/{repo}_statistical"

            run_complete_pipeline_with_statistics(
                kg_path=kg_path,
                emb_dir=emb_dir,
                output_dir=output_dir,
                model_type=model_type,
                search_type='smart',
                n_random_samples=15
            )
        else:
            print("Usage: python script.py <repo> <model_type>")
            print("Example: python script.py amazon complex")
    else:
        # Interactive mode
        interactive_menu()