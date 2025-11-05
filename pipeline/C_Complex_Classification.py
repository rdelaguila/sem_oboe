import pandas as pd
import numpy as np
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from pykeen.evaluation import RankBasedEvaluator
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
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

warnings.filterwarnings('ignore')


class ComplExOptimizer:
    """
    Optimized ComplEx training with hyperparameter search for better MRR
    """

    def __init__(self, kg_path: str, emb_dir: str, output_dir: str = "../data/model_output/complex_optim"):
        self.kg_csv_path = kg_path
        self.emb_dir = emb_dir
        self.output_dir = output_dir
        self.model_type = 'complex'
        self.model = None
        self.entity_to_id = None
        self.relation_to_id = None
        self.best_embedding_dim = None
        self.best_hyperparams = None

        # Create directories
        os.makedirs(emb_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # Setup output files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_file = os.path.join(output_dir, f"complex_optimization_results_{timestamp}.json")
        self.metrics_file = os.path.join(output_dir, f"complex_optimization_metrics_{timestamp}.txt")
        self.best_model_file = os.path.join(emb_dir, 'complex_best_model.pkl')
        self.best_mappings_file = os.path.join(emb_dir, 'complex_best_mappings.pkl')

    def prepare_triples_factory(self) -> Tuple[TriplesFactory, TriplesFactory]:
        """Prepare triples factory for ComplEx training"""
        print("\nPreparing triples factory for ComplEx...")
        dftrain = pd.read_csv(self.kg_csv_path)
        dftest = pd.read_csv(self.kg_csv_path.replace('sampled_train','sampled_test'))
        triplestr = dftrain[['subject', 'relation', 'object']].values
        tripletest = dftest[['subject', 'relation', 'object']].values

        print("Creating inverse triples (recommended for ComplEx)")

        # Create triples factory with inverse triples
        training_tf = TriplesFactory.from_labeled_triples(
            triplestr,
            create_inverse_triples=True,  # Always True for ComplEx
        )
        testing_tf = TriplesFactory.from_labeled_triples(
            triplestr,
            create_inverse_triples=True,  # Always True for ComplEx
        )
       # from pykeen.triples.utils import split

        #training_tf, testing_tf = tf.split([0.8, 0.2], random_state=42)


        #print(f"Total triples: {len(triples)}")
        #print(f"Total triples (with inverse): {len(tf.triples)}")
        print(f"Training triples: {len(training_tf.triples)}")
        print(f"Testing triples: {len(testing_tf.triples)}")
        print(f"Entities: {training_tf.num_entities}")
        print(f"Relations: {training_tf.num_relations}")

        # Diagnostics
        print(f"\nData distribution:")
        print(f"  Train/Test split: 80/20")
        print(f"  Avg triples per entity: {len(training_tf.triples) / training_tf.num_entities:.2f}")
        print(f"  Avg triples per relation: {len(training_tf.triples) / training_tf.num_relations:.2f}")

        return training_tf, testing_tf

    def get_hyperparameter_grid(self) -> Dict[str, List]:
        """Define hyperparameter search space for ComplEx"""
        return {
            'embedding_dim': [64, 100, 128, 150, 200],  # Smaller dimensions to reduce overfitting
            'learning_rate': [0.0001, 0.0005, 0.001, 0.005, 0.01],
            'num_epochs': [200, 300, 400, 500],
            'batch_size': [128, 256, 512],
            'regularizer_weight': [0.0001, 0.001, 0.01, 0.1],  # L2 regularization weight
        }

    def train_complex_single_config(self, training_tf: TriplesFactory, testing_tf: TriplesFactory,
                                    embedding_dim: int, learning_rate: float, num_epochs: int,
                                    batch_size: int, regularizer_weight: float) -> Tuple[Any, Dict[str, Any]]:
        """Train ComplEx with specific hyperparameters"""
        #"mps" if torch.backends.mps.is_available() else "cpu"
        config_str = (f"dim={embedding_dim}, lr={learning_rate}, epochs={num_epochs}, "
                      f"batch={batch_size}, reg={regularizer_weight}")
        print(f"\n{'=' * 80}")
        print(f"Training ComplEx: {config_str}")
        print(f"{'=' * 80}")

        # ComplEx-specific model kwargs
        model_kwargs = {
            'embedding_dim': embedding_dim,
            'regularizer': 'lp',
            'regularizer_kwargs': dict(p=2, weight=regularizer_weight),  # L2 regularization
        }

        # Train model with PyKEEN
        try:
            result = pipeline(
                training=training_tf,
                testing=testing_tf,
                model='complex',

                model_kwargs=model_kwargs,
                optimizer='adam',
                optimizer_kwargs=dict(lr=learning_rate),
                loss='nssa',  # Negative Sampling Self-Adversarial loss (better for ComplEx)
                loss_kwargs=dict(adversarial_temperature=1.0),
                training_kwargs=dict(
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                    use_tqdm_batch=False,
                    use_tqdm=True,
                ),
                evaluator_kwargs=dict(filtered=True),
                evaluation_kwargs=dict(batch_size=256),
                random_seed=42,
            )

            # Extract evaluation metrics
            metrics = {
                'embedding_dim': embedding_dim,
                'learning_rate': learning_rate,
                'num_epochs': num_epochs,
                'batch_size': batch_size,
                'regularizer_weight': regularizer_weight,
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
                metrics['score'] = metrics['evaluation']['mrr']  # Use MRR as primary score

                print(f"\nResults for {config_str}:")
                print(f"  Hits@1: {metrics['evaluation']['hits_at_1']:.4f}")
                print(f"  Hits@3: {metrics['evaluation']['hits_at_3']:.4f}")
                print(f"  Hits@10: {metrics['evaluation']['hits_at_10']:.4f}")
                print(f"  MRR: {metrics['evaluation']['mrr']:.4f}")
                print(f"  MR: {metrics['evaluation']['mr']:.2f}")
            else:
                metrics['score'] = 0.0

            return result.model, metrics

        except Exception as e:
            print(f"ERROR training with {config_str}: {e}")
            metrics = {
                'embedding_dim': embedding_dim,
                'learning_rate': learning_rate,
                'num_epochs': num_epochs,
                'batch_size': batch_size,
                'regularizer_weight': regularizer_weight,
                'score': 0.0,
                'error': str(e)
            }
            return None, metrics

    def optimize_hyperparameters(self, training_tf: TriplesFactory, testing_tf: TriplesFactory,
                                 search_type: str = 'random', n_random_samples: int = 20) -> Dict[str, Any]:
        """Perform hyperparameter optimization"""

        print("\n" + "=" * 80)
        print("STARTING HYPERPARAMETER OPTIMIZATION FOR COMPLEX")
        print("=" * 80)

        param_grid = self.get_hyperparameter_grid()

        if search_type == 'grid':
            # Full grid search
            print("\nPerforming FULL GRID SEARCH")
            print("This will take a long time...")

            param_combinations = list(product(
                param_grid['embedding_dim'],
                param_grid['learning_rate'],
                param_grid['num_epochs'],
                param_grid['batch_size'],
                param_grid['regularizer_weight']
            ))

            print(f"Total combinations to try: {len(param_combinations)}")

        elif search_type == 'random':
            # Random search
            print(f"\nPerforming RANDOM SEARCH with {n_random_samples} samples")

            # Generate random combinations
            param_combinations = []
            np.random.seed(42)
            for _ in range(n_random_samples):
                combo = (
                    np.random.choice(param_grid['embedding_dim']),
                    np.random.choice(param_grid['learning_rate']),
                    np.random.choice(param_grid['num_epochs']),
                    np.random.choice(param_grid['batch_size']),
                    np.random.choice(param_grid['regularizer_weight']),
                )
                param_combinations.append(combo)

        else:
            # Smart search - start with good defaults, then explore
            print("\nPerforming SMART SEARCH")
            param_combinations = self._get_smart_search_combinations()

        print(f"Trying {len(param_combinations)} configurations\n")

        best_score = -float('inf')
        best_model = None
        best_metrics = None
        all_results = []

        for i, (emb_dim, lr, epochs, batch, reg_weight) in enumerate(param_combinations, 1):
            print(f"\n[{i}/{len(param_combinations)}] Testing configuration...")

            model, metrics = self.train_complex_single_config(
                training_tf, testing_tf, emb_dim, lr, epochs, batch, reg_weight
            )

            all_results.append(metrics)
            score = metrics.get('score', 0.0)

            if score > best_score and model is not None:
                best_score = score
                best_model = model
                best_metrics = metrics
                self.best_embedding_dim = emb_dim
                self.best_hyperparams = {
                    'embedding_dim': emb_dim,
                    'learning_rate': lr,
                    'num_epochs': epochs,
                    'batch_size': batch,
                    'regularizer_weight': reg_weight,
                }
                print(f"  ★ NEW BEST MODEL! MRR: {best_score:.4f}")

        # Store best model
        self.model = best_model
        self.entity_to_id = training_tf.entity_to_id
        self.relation_to_id = training_tf.relation_to_id

        # Add optimization results
        best_metrics['all_results'] = all_results
        best_metrics['best_score'] = best_score
        best_metrics['best_hyperparams'] = self.best_hyperparams
        best_metrics['search_type'] = search_type
        best_metrics['total_configurations_tried'] = len(param_combinations)

        print("\n" + "=" * 80)
        print("HYPERPARAMETER OPTIMIZATION COMPLETED")
        print("=" * 80)
        print(f"Best configuration found:")
        for key, value in self.best_hyperparams.items():
            print(f"  {key}: {value}")
        print(f"Best MRR: {best_score:.4f}")

        return best_metrics

    def _get_smart_search_combinations(self) -> List[Tuple]:
        """Get smart search combinations - start with good defaults, then explore"""
        combinations = []

        combinations = [
            # Aumentar épocas con la mejor config
            #(128, 0.02, 300, 256, 0.01),  # ⭐⭐⭐

            (64, 0.02, 300, 256, 0.010),  # Baseline (ya tienes: MRR=0.82)
            (64, 0.02, 300, 256, 0.009),  # Menos reg: puede mejorar
            (64, 0.02, 300, 256, 0.011),  # Más reg: más generalización

            (128, 0.02, 300, 256, 0.010),  # Baseline (ya tienes: MRR=0.82)
            (128, 0.02, 300, 256, 0.009),  # Menos reg: puede mejorar
            (128, 0.02, 300, 256, 0.011),  # Más reg: más generalización

            # dim=150 con config óptima
            (150, 0.02, 300, 256, 0.01),
            #(150, 0.02, 400, 256, 0.01),

            # Ajustar regularización con más épocas
            #(200, 0.02, 250, 256, 0.005),
            (200, 0.02, 300, 256, 0.01),
        ]

        print("Estimación: ~3-4 min por config = 45-60 min total")
        # Remove duplicates
        combinations = list(set(combinations))

        return combinations

    def save_best_model_and_results(self, optimization_results: Dict[str, Any]) -> str:
        """Save the best ComplEx model and optimization results"""

        # Save model
        torch.save(self.model, self.best_model_file)

        # Save mappings with embedding dimension
        mappings = {
            'entity_to_id': self.entity_to_id,
            'relation_to_id': self.relation_to_id,
            'model_type': 'complex',
            'embedding_dim': self.best_embedding_dim,
            'best_hyperparams': self.best_hyperparams,
        }
        joblib.dump(mappings, self.best_mappings_file)

        # Save optimization results (basic info only for JSON)
        basic_info = {
            'timestamp': datetime.now().isoformat(),
            'model_path': self.best_model_file,
            'mappings_path': self.best_mappings_file,
            'model_type': 'complex',
            'best_embedding_dim': self.best_embedding_dim,
            'best_hyperparams': self.best_hyperparams,
            'best_mrr': optimization_results['best_score'],
            'search_type': optimization_results['search_type'],
            'total_configs_tried': optimization_results['total_configurations_tried'],
        }

        with open(self.results_file, 'w') as f:
            json.dump(basic_info, f, indent=2)

        # Save detailed metrics to text file
        with open(self.metrics_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("COMPLEX HYPERPARAMETER OPTIMIZATION RESULTS\n")
            f.write("=" * 80 + "\n\n")

            f.write("BEST CONFIGURATION:\n")
            f.write("-" * 80 + "\n")
            for key, value in self.best_hyperparams.items():
                f.write(f"{key}: {value}\n")
            f.write(f"\nBest MRR: {optimization_results['best_score']:.4f}\n")

            if 'evaluation' in optimization_results:
                f.write("\nBest Model Evaluation Metrics:\n")
                f.write("-" * 80 + "\n")
                eval_metrics = optimization_results['evaluation']
                f.write(f"Hits@1: {eval_metrics['hits_at_1']:.4f}\n")
                f.write(f"Hits@3: {eval_metrics['hits_at_3']:.4f}\n")
                f.write(f"Hits@10: {eval_metrics['hits_at_10']:.4f}\n")
                f.write(f"MRR: {eval_metrics['mrr']:.4f}\n")
                f.write(f"MR: {eval_metrics['mr']:.2f}\n")

            f.write(f"\n\nSearch Strategy: {optimization_results['search_type']}\n")
            f.write(f"Total Configurations Tried: {optimization_results['total_configurations_tried']}\n\n")

            f.write("\nALL CONFIGURATIONS TESTED:\n")
            f.write("=" * 80 + "\n")

            # Sort by score
            sorted_results = sorted(optimization_results['all_results'],
                                    key=lambda x: x.get('score', 0.0), reverse=True)

            for i, result in enumerate(sorted_results, 1):
                f.write(f"\n#{i} - Score (MRR): {result.get('score', 0.0):.4f}\n")
                f.write(f"  Embedding Dim: {result.get('embedding_dim', 'N/A')}\n")
                f.write(f"  Learning Rate: {result.get('learning_rate', 'N/A')}\n")
                f.write(f"  Epochs: {result.get('num_epochs', 'N/A')}\n")
                f.write(f"  Batch Size: {result.get('batch_size', 'N/A')}\n")
                f.write(f"  Regularizer Weight: {result.get('regularizer_weight', 'N/A')}\n")

                if 'evaluation' in result:
                    eval_m = result['evaluation']
                    f.write(f"  Hits@10: {eval_m['hits_at_10']:.4f}\n")

                if 'error' in result:
                    f.write(f"  ERROR: {result['error']}\n")

        print(f"\n{'=' * 80}")
        print("Results saved:")
        print(f"  Model: {self.best_model_file}")
        print(f"  Mappings: {self.best_mappings_file}")
        print(f"  Results JSON: {self.results_file}")
        print(f"  Metrics TXT: {self.metrics_file}")
        print("=" * 80)

        return self.best_model_file

    def run_optimization(self, search_type: str = 'smart', n_random_samples: int = 15) -> str:
        """Run complete ComplEx optimization pipeline"""
        print("=" * 80)
        print("STARTING COMPLEX OPTIMIZATION PIPELINE")
        print("=" * 80)

        # Prepare data
        training_tf, testing_tf = self.prepare_triples_factory()

        # Optimize hyperparameters
        optimization_results = self.optimize_hyperparameters(
            training_tf, testing_tf,
            search_type=search_type,
            n_random_samples=n_random_samples
        )

        # Save best model and results
        model_path = self.save_best_model_and_results(optimization_results)

        print("\n" + "=" * 80)
        print("COMPLEX OPTIMIZATION COMPLETED!")
        print("=" * 80)

        return model_path


class TopicClassifier:
    """
    Phase 2: Topic classification using optimized ComplEx embeddings
    (Same as before, no changes)
    """

    def __init__(self, kge_csv_path: str, model_path: str, mappings_path: str,
                 output_dir: str = "../data/model_output/complex_optim"):
        self.kge_csv_path = kge_csv_path
        self.model_path = model_path
        self.mappings_path = mappings_path
        self.output_dir = output_dir

        self.kge_model = None
        self.entity_to_id = None
        self.relation_to_id = None
        self.model_type = None
        self.embedding_dim = None
        self.is_binary_classification = None

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Setup output files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_file = os.path.join(output_dir, f"classification_results_{timestamp}.json")
        self.metrics_file = os.path.join(output_dir, f"classification_metrics_{timestamp}.txt")
        self.best_model_file = os.path.join(output_dir, f"xgboost_best_model_{timestamp}.pkl")

    def load_kge_model_and_mappings(self):
        """Load trained KGE model and entity/relation mappings"""
        print("\nLoading ComplEx model and mappings...")
        self.kge_model = torch.load(self.model_path, weights_only=False)
        mappings = joblib.load(self.mappings_path)

        self.entity_to_id = mappings['entity_to_id']
        self.relation_to_id = mappings['relation_to_id']
        self.model_type = mappings['model_type']
        self.embedding_dim = mappings.get('embedding_dim', 150)

        print(f"Model type: {self.model_type}")
        print(f"Embedding dimension: {self.embedding_dim}")
        print(f"Entities: {len(self.entity_to_id)}")
        print(f"Relations: {len(self.relation_to_id)}")

        if 'best_hyperparams' in mappings:
            print("\nOptimized hyperparameters:")
            for key, value in mappings['best_hyperparams'].items():
                print(f"  {key}: {value}")

    def load_classification_data(self) -> pd.DataFrame:
        """Load and prepare classification data"""
        print("\nLoading classification data...")
        df = pd.read_csv(self.kge_csv_path)

        # Determine if binary or multiclass
        unique_topics = df['new_topic'].nunique()
        self.is_binary_classification = unique_topics == 2

        print(f"Total samples: {len(df)}")
        print(f"Unique topics: {unique_topics}")
        print(f"Classification type: {'Binary' if self.is_binary_classification else 'Multiclass'}")
        print("\nTopic distribution:")
        print(df['new_topic'].value_counts())

        return df

    def get_entity_embedding(self, entity: str) -> np.ndarray:
        """Get embedding for an entity"""
        if entity not in self.entity_to_id:
            # For ComplEx, return zero vector with doubled dimension
            if self.model_type == 'complex':
                return np.zeros(self.embedding_dim * 2)
            else:
                return np.zeros(self.embedding_dim)

        entity_id = self.entity_to_id[entity]
        entity_tensor = torch.tensor([entity_id], dtype=torch.long)

        with torch.no_grad():
            embedding = self.kge_model.entity_representations[0](entity_tensor)

        embedding_np = embedding.cpu().numpy().flatten()

        # Handle complex embeddings - convert to real for XGBoost
        if np.iscomplexobj(embedding_np):
            embedding_real = np.real(embedding_np)
            embedding_imag = np.imag(embedding_np)
            embedding_np = np.concatenate([embedding_real, embedding_imag])

        return embedding_np

    def create_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create features from KGE embeddings"""
        print("\nCreating features from ComplEx embeddings...")

        X_list = []
        y_list = []

        for idx, row in df.iterrows():
            subject_emb = self.get_entity_embedding(row['subject'])
            object_emb = self.get_entity_embedding(row['object'])

            # Concatenate embeddings
            combined_emb = np.concatenate([subject_emb, object_emb])
            X_list.append(combined_emb)
            y_list.append(row['new_topic'])

            if (idx + 1) % 1000 == 0:
                print(f"Processed {idx + 1}/{len(df)} samples...")

        X = np.array(X_list)
        y = np.array(y_list)

        # Calculate actual feature dimension per entity
        actual_feature_dim_per_entity = X.shape[1] // 2

        print(f"\nFeature matrix shape: {X.shape}")
        print(f"Target vector shape: {y.shape}")
        print(f"Base embedding dimension: {self.embedding_dim}")
        print(f"Actual feature dimension per entity: {actual_feature_dim_per_entity}")
        print(f"Total feature dimension: {X.shape[1]}")

        # Note if complex embeddings were used
        if actual_feature_dim_per_entity == self.embedding_dim * 2:
            print("Note: Complex embeddings detected - using concatenated real and imaginary parts")
        elif actual_feature_dim_per_entity == self.embedding_dim:
            print("Note: Real-valued embeddings used")

        return X, y

    def calculate_per_class_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                    y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Calculate detailed per-class metrics"""
        classes = np.unique(y_true)
        per_class_metrics = {}

        for cls in classes:
            cls_mask = (y_true == cls)
            cls_accuracy = accuracy_score(y_true[cls_mask], y_pred[cls_mask])
            per_class_metrics[str(cls)] = {
                'accuracy': float(cls_accuracy),
                'error': float(1 - cls_accuracy),
                'samples': int(np.sum(cls_mask))
            }

        mean_per_class_error = np.mean([m['error'] for m in per_class_metrics.values()])

        return {
            'per_class': per_class_metrics,
            'mean_per_class_error': float(mean_per_class_error)
        }

    def calculate_auc_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Calculate AUC metrics for binary or multiclass classification"""
        auc_metrics = {}

        try:
            if self.is_binary_classification:
                # Binary classification
                if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == 2:
                    y_scores = y_pred_proba[:, 1]
                else:
                    y_scores = y_pred_proba

                fpr, tpr, _ = roc_curve(y_true, y_scores)
                roc_auc = auc(fpr, tpr)
                auc_metrics['roc_auc'] = float(roc_auc)
                auc_metrics['fpr'] = fpr.tolist()
                auc_metrics['tpr'] = tpr.tolist()
            else:
                # Multiclass classification
                classes = np.unique(y_true)
                if len(classes) > 2:
                    try:
                        roc_auc_ovr = roc_auc_score(y_true, y_pred_proba,
                                                    multi_class='ovr', average='macro')
                        auc_metrics['roc_auc_macro'] = float(roc_auc_ovr)
                    except Exception as e:
                        print(f"Warning: Could not calculate macro AUC: {e}")
                        auc_metrics['roc_auc_macro'] = None
        except Exception as e:
            print(f"Warning: Could not calculate AUC metrics: {e}")

        return auc_metrics

    def evaluate_model(self, model: XGBClassifier, X_test: np.ndarray,
                       y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate model performance"""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        logloss = log_loss(y_test, y_pred_proba)

        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Per-class metrics
        per_class_info = self.calculate_per_class_metrics(y_test, y_pred, y_pred_proba)

        # AUC metrics
        auc_metrics = self.calculate_auc_metrics(y_test, y_pred_proba)

        metrics = {
            'accuracy': float(accuracy),
            'logloss': float(logloss),
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'per_class_metrics': per_class_info['per_class'],
            'mean_per_class_error': per_class_info['mean_per_class_error'],
            'auc_metrics': auc_metrics
        }

        return metrics

    def save_classification_results(self, metrics: Dict[str, Any],
                                    best_params: Dict[str, Any],
                                    cv_results: Dict[str, Any]):
        """Save classification results"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'model_type': 'XGBoost',
            'kge_model_type': self.model_type,
            'embedding_dim': self.embedding_dim,
            'classification_type': 'binary' if self.is_binary_classification else 'multiclass',
            'best_params': best_params,
            'cv_results': cv_results,
            'test_metrics': metrics
        }

        with open(self.results_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Save human-readable metrics
        with open(self.metrics_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("TOPIC CLASSIFICATION RESULTS\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"KGE Model Type: {self.model_type}\n")
            f.write(f"Embedding Dimension: {self.embedding_dim}\n")
            f.write(f"Classification Type: {'Binary' if self.is_binary_classification else 'Multiclass'}\n")
            f.write(f"Test Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Test Log Loss: {metrics['logloss']:.4f}\n")
            f.write(f"Mean Per-Class Error: {metrics['mean_per_class_error']:.4f}\n\n")

            if 'roc_auc' in metrics['auc_metrics']:
                f.write(f"ROC AUC: {metrics['auc_metrics']['roc_auc']:.4f}\n\n")
            elif 'roc_auc_macro' in metrics['auc_metrics'] and metrics['auc_metrics']['roc_auc_macro']:
                f.write(f"ROC AUC (Macro): {metrics['auc_metrics']['roc_auc_macro']:.4f}\n\n")

            f.write("Best Hyperparameters:\n")
            f.write("-" * 60 + "\n")
            for param, value in best_params.items():
                f.write(f"{param}: {value}\n")

            f.write("\n" + "=" * 60 + "\n")
            f.write("CLASSIFICATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            for label, metrics_dict in metrics['classification_report'].items():
                if isinstance(metrics_dict, dict):
                    f.write(f"\n{label}:\n")
                    for metric_name, value in metrics_dict.items():
                        f.write(f"  {metric_name}: {value:.4f}\n")

        print(f"\nResults saved to: {self.results_file}")
        print(f"Metrics saved to: {self.metrics_file}")
        print(f"Best model saved to: {self.best_model_file}")

    def run_classification_with_grid_search(self, X: np.ndarray, y: np.ndarray,
                                            use_random_search: bool = True,
                                            n_iter: int = 20,
                                            cv: int = 3) -> Tuple[XGBClassifier, Dict[str, Any]]:
        """Run classification with hyperparameter tuning"""
        print("\nPreparing classification with hyperparameter tuning...")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")

        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2]
        }

        # Base model
        base_model = XGBClassifier(
            objective='binary:logistic' if self.is_binary_classification else 'multi:softprob',
            eval_metric='logloss',
            random_state=42,
            use_label_encoder=False
        )

        # Grid or Random Search
        if use_random_search:
            print(f"Using RandomizedSearchCV with {n_iter} iterations and {cv}-fold CV...")
            search = RandomizedSearchCV(
                base_model,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring='neg_log_loss',
                n_jobs=-1,
                verbose=2,
                random_state=42
            )
        else:
            print(f"Using GridSearchCV with {cv}-fold CV...")
            search = GridSearchCV(
                base_model,
                param_grid=param_grid,
                cv=cv,
                scoring='neg_log_loss',
                n_jobs=-1,
                verbose=2
            )

        # Fit
        search.fit(X_train, y_train)

        # Best model
        best_model = search.best_estimator_

        print(f"\nBest parameters found:")
        for param, value in search.best_params_.items():
            print(f"  {param}: {value}")

        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_metrics = self.evaluate_model(best_model, X_test, y_test)

        print(f"\nTest Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test Log Loss: {test_metrics['logloss']:.4f}")
        print(f"Mean Per-Class Error: {test_metrics['mean_per_class_error']:.4f}")

        # CV results
        cv_results = {
            'best_score': float(search.best_score_),
            'best_params': search.best_params_
        }

        # Save results
        self.save_classification_results(test_metrics, search.best_params_, cv_results)

        # Save best model
        joblib.dump(best_model, self.best_model_file)

        return best_model, test_metrics

    def run_topic_classification(self, use_random_search: bool = True,
                                 n_iter: int = 20, cv: int = 3) -> Tuple[XGBClassifier, Dict[str, Any]]:
        """Run complete topic classification phase"""
        print("=" * 60)
        print("STARTING PHASE 2: TOPIC CLASSIFICATION")
        print("=" * 60)

        # Load ComplEx model and mappings
        self.load_kge_model_and_mappings()

        # Load classification data
        df = self.load_classification_data()

        # Create features
        X, y = self.create_features(df)

        # Run classification with grid search
        best_model, metrics = self.run_classification_with_grid_search(
            X, y, use_random_search=use_random_search, n_iter=n_iter, cv=cv
        )

        print("\n" + "=" * 60)
        print("PHASE 2 COMPLETED: TOPIC CLASSIFICATION FINISHED")
        print("=" * 60)

        # Summary results
        print(f"Classification Type: {'Binary' if self.is_binary_classification else 'Multiclass'}")
        print(f"Final Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"Final Test Log Loss: {metrics['logloss']:.4f}")
        print(f"Final Mean Per-Class Error: {metrics['mean_per_class_error']:.4f}")

        if 'roc_auc' in metrics['auc_metrics']:
            print(f"Final ROC AUC: {metrics['auc_metrics']['roc_auc']:.4f}")
        elif 'roc_auc_macro' in metrics['auc_metrics'] and metrics['auc_metrics']['roc_auc_macro']:
            print(f"Final ROC AUC (Macro): {metrics['auc_metrics']['roc_auc_macro']:.4f}")

        return best_model, metrics


# Main execution functions
def run_complex_optimization(kg_path: str, emb_dir: str, output_dir: str,
                             search_type: str = 'smart', n_random_samples: int = 15) -> str:
    """
    Execute ComplEx hyperparameter optimization

    Args:
        kg_path: Path to knowledge graph CSV file
        emb_dir: Directory for embeddings storage
        output_dir: Directory for output files
        search_type: 'grid', 'random', or 'smart'
        n_random_samples: Number of samples for random search

    Returns:
        Path to the best trained model
    """
    optimizer = ComplExOptimizer(kg_path=kg_path, emb_dir=emb_dir, output_dir=output_dir)
    return optimizer.run_optimization(search_type=search_type, n_random_samples=n_random_samples)


def run_classification_phase(kg_path: str, model_path: str, mappings_path: str,
                             output_dir: str,
                             use_random_search: bool = True, n_iter: int = 20, cv: int = 3) -> Tuple[
    XGBClassifier, Dict[str, Any]]:
    """
    Execute Phase 2: Topic Classification

    Args:
        kg_path: Path to csv triples files
        model_path: Path to trained ComplEx model
        mappings_path: Path to entity/relation mappings
        output_dir: Directory for output files
        use_random_search: Use RandomizedSearchCV if True, GridSearchCV if False
        n_iter: Number of iterations for RandomizedSearchCV
        cv: Number of cross-validation folds

    Returns:
        Tuple of (best_model, metrics)
    """
    classifier = TopicClassifier(
        kge_csv_path=kg_path,
        model_path=model_path,
        mappings_path=mappings_path,
        output_dir=output_dir
    )

    return classifier.run_topic_classification(
        use_random_search=use_random_search,
        n_iter=n_iter,
        cv=cv
    )


def run_complete_pipeline(kg_path: str, emb_dir: str, output_dir: str,
                          search_type: str = 'smart', n_random_samples: int = 15,
                          use_random_search: bool = True, n_iter: int = 20, cv: int = 3) -> Tuple[
    XGBClassifier, Dict[str, Any]]:
    """
    Execute both phases sequentially

    Args:
        kg_path: Path to knowledge graph CSV file
        emb_dir: Directory for embeddings storage
        output_dir: Directory for output files
        search_type: 'grid', 'random', or 'smart' for ComplEx optimization
        n_random_samples: Number of samples for random search
        use_random_search: Use RandomizedSearchCV for classification
        n_iter: Number of iterations for RandomizedSearchCV
        cv: Number of cross-validation folds

    Returns:
        Tuple of (best_model, metrics)
    """
    print("STARTING COMPLETE TWO-PHASE PIPELINE WITH COMPLEX OPTIMIZATION")
    print("=" * 80)

    # Phase 1: ComplEx Optimization
    model_path = run_complex_optimization(
        kg_path=kg_path,
        emb_dir=emb_dir,
        output_dir=output_dir,
        search_type=search_type,
        n_random_samples=n_random_samples
    )

    # Derive mappings path from model path
    mappings_path = model_path.replace('_model.pkl', '_mappings.pkl')

    print(f"\nPHASE 1 COMPLETE. Starting Phase 2...")
    print(f"Using model: {model_path}")
    print(f"Using mappings: {mappings_path}")

    # Phase 2: Classification
    best_model, metrics = run_classification_phase(
        kg_path=kg_path,
        model_path=model_path,
        mappings_path=mappings_path,
        output_dir=output_dir,
        use_random_search=use_random_search,
        n_iter=n_iter,
        cv=cv
    )

    print("\n" + "=" * 80)
    print("COMPLETE PIPELINE FINISHED SUCCESSFULLY!")
    print("=" * 80)

    return best_model, metrics


def interactive_menu():
    """Interactive menu for selecting operations"""
    print("\n" + "=" * 80)
    print("COMPLEX OPTIMIZATION MODULE - INTERACTIVE MENU")
    print("=" * 80)

    # Repository selection
    print("\nAvailable repositories:")
    print("1. Amazon")
    print("2. BBC")
    print("3. reuters")
    print("=" * 80)

    repo_choice = input("\nSelect repository (1-3): ").strip()

    if repo_choice == '1':
        repo = 'amazon'
        dataset_name = 'dataset_triplet_amazon_new_simplificado.csv'
    elif repo_choice == '2':
        repo = 'bbc'
        dataset_name = 'dataset_triplet_bbc_new_simplificado.csv'
    elif repo_choice == '3':
        repo = 'reuters_activities'
        dataset_name = 'dataset_triplet_reuters_activities_new_simplificado.csv'
    else:
        print("Invalid repository choice. Exiting...")
        sys.exit(1)

    print(f"\nSelected repository: {repo.upper()}")

    print("\nSelect an operation:")
    print("1. Phase 1: ComplEx Optimization Only")
    print("2. Phase 2: Classification Only (requires optimized ComplEx model)")
    print("3. Complete Pipeline (Phase 1 + Phase 2)")
    print("4. Exit")
    print("=" * 80)

    choice = input("\nEnter your choice (1-4): ").strip()

    if choice == '4':
        print("Exiting...")
        sys.exit(0)

    # Define paths based on repository
    base_path = "../data"
    kg_path = f"{base_path}/triples_raw/{repo}/{dataset_name}"
    emb_dir = f"{base_path}/triples_emb/{repo}"
    model_path = f"{emb_dir}/complex_best_model.pkl"
    mappings_path = f"{emb_dir}/complex_best_mappings.pkl"

    # Output directory
    output_dir = f"{base_path}/model_output/{repo}_complex_optim"

    if choice == '1':
        # Phase 1: ComplEx Optimization
        print("\n--- Phase 1: ComplEx Optimization ---")
        print("Search strategies:")
        print("  1. Smart search (recommended, ~15-20 configs)")
        print("  2. Random search (specify number)")
        print("  3. Grid search (exhaustive, VERY slow)")

        search_choice = input("\nSelect search strategy (1-3, default: 1): ").strip() or "1"

        if search_choice == '1':
            search_type = 'smart'
            n_samples = 0
        elif search_choice == '2':
            search_type = 'random'
            n_samples = int(input("Number of random configurations to try (default: 20): ").strip() or "20")
        else:
            search_type = 'grid'
            n_samples = 0
            print("WARNING: Grid search will try 100+ configurations and take HOURS!")
            confirm = input("Continue? (yes/no): ").strip().lower()
            if confirm != 'yes':
                print("Aborted.")
                sys.exit(0)

        model_path = run_complex_optimization(
            kg_path=kg_path,
            emb_dir=emb_dir,
            output_dir=output_dir,
            search_type=search_type,
            n_random_samples=n_samples
        )
        print(f"\nPhase 1 completed. Best model saved to: {model_path}")

    elif choice == '2':
        # Phase 2: Classification
        print("\n--- Phase 2: Classification ---")
        use_random_search_input = input("Use RandomizedSearchCV? (y/n, default: y): ").strip().lower()
        use_random_search = use_random_search_input != 'n'

        if use_random_search:
            n_iter = int(input("Number of iterations for RandomizedSearchCV (default: 20): ").strip() or "20")
        else:
            n_iter = 20

        cv = int(input("Number of cross-validation folds (default: 3): ").strip() or "3")

        best_model, metrics = run_classification_phase(
            kg_path=kg_path,
            model_path=model_path,
            mappings_path=mappings_path,
            output_dir=output_dir,
            use_random_search=use_random_search,
            n_iter=n_iter,
            cv=cv
        )
        print(f"\nPhase 2 completed. Best accuracy: {metrics['accuracy']:.4f}")

    elif choice == '3':
        # Complete Pipeline
        print("\n--- Complete Pipeline ---")
        print("Search strategies for ComplEx:")
        print("  1. Smart search (recommended)")
        print("  2. Random search")

        search_choice = input("\nSelect search strategy (1-2, default: 1): ").strip() or "1"

        if search_choice == '2':
            search_type = 'random'
            n_samples = int(input("Number of random configurations (default: 15): ").strip() or "15")
        else:
            search_type = 'smart'
            n_samples = 15

        use_random_search_input = input(
            "\nUse RandomizedSearchCV for classification? (y/n, default: y): ").strip().lower()
        use_random_search = use_random_search_input != 'n'

        if use_random_search:
            n_iter = int(input("Number of iterations for classification (default: 20): ").strip() or "20")
        else:
            n_iter = 20

        cv = int(input("Number of cross-validation folds (default: 3): ").strip() or "3")

        best_model, metrics = run_complete_pipeline(
            kg_path=kg_path,
            emb_dir=emb_dir,
            output_dir=output_dir,
            search_type=search_type,
            n_random_samples=n_samples,
            use_random_search=use_random_search,
            n_iter=n_iter,
            cv=cv
        )
        print(f"\nComplete pipeline finished. Final accuracy: {metrics['accuracy']:.4f}")

    else:
        print("Invalid choice. Please run the script again.")
        sys.exit(1)


if __name__ == "__main__":
    # Check if running with command line arguments (programmatic mode)
    if len(sys.argv) > 1:
        # Programmatic mode
        print("ComplEx Optimization Module - Programmatic Mode")
        print("=" * 80)

        if len(sys.argv) >= 3:
            phase = sys.argv[1]
            repo = sys.argv[2].lower()

            # Validate repository
            if repo not in ['amazon', 'bbc','reuters']:
                print(f"Error: Invalid repository '{repo}'")
                print("Available repositories: amazon, bbc")
                sys.exit(1)

            # Set dataset name based on repository
            if repo == 'amazon':
                dataset_name = 'dataset_triplet_reuters_activities_new_simplificado.csv'
            elif repo == 'bbc':
                dataset_name = 'dataset_triplet_bbc_new_simplificado.csv'

            base_path = "../data"
            kg_path = f"{base_path}/triples_raw/{repo}/{dataset_name}"
            emb_dir = f"{base_path}/triples_emb/{repo}"
            model_path = f"{emb_dir}/complex_best_model.pkl"
            mappings_path = f"{emb_dir}/complex_best_mappings.pkl"
            output_dir = f"{base_path}/model_output/{repo}_complex_optim"

            print(f"Repository: {repo.upper()}")
            print(f"Dataset: {dataset_name}")
            print(f"Output directory: {output_dir}")
            print("=" * 80)

            if phase == "optimize":
                run_complex_optimization(
                    kg_path=kg_path,
                    emb_dir=emb_dir,
                    output_dir=output_dir,
                    search_type='smart',
                    n_random_samples=15
                )
            elif phase == "classify":
                run_classification_phase(
                    kg_path=kg_path,
                    model_path=model_path,
                    mappings_path=mappings_path,
                    output_dir=output_dir
                )
            elif phase == "complete":
                run_complete_pipeline(
                    kg_path=kg_path,
                    emb_dir=emb_dir,
                    output_dir=output_dir,
                    search_type='smart',
                    n_random_samples=15
                )
            else:
                print(f"Unknown phase: {phase}")
                print("Available phases: optimize, classify, complete")
        else:
            print("Usage: python script.py <phase> <repo_name>")
            print("Phases: optimize, classify, complete")
            print("Repositories: amazon, bbc")
    else:
        # Interactive mode
        print("ComplEx Optimization Module - Interactive Mode")
        print("This module optimizes ComplEx hyperparameters for better MRR")
        print("while maintaining the same classification pipeline.")

        interactive_menu()