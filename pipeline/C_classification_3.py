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

warnings.filterwarnings('ignore')


class KGEModelTrainer:
    """
    Phase 1: Knowledge Graph Embeddings (KGE) model training
    """

    AVAILABLE_MODELS = {
        'transe': 'TransE',
        'convkb': 'ConvKB',
        'complex': 'ComplEx',
        'distmult': 'DistMult',
        'rotate': 'RotatE'
    }

    def __init__(self, kg_path: str, emb_dir: str,
                 output_dir: str = "data/model_output/amazon", model_type: str = None,
                 embedding_dims: List[int] = None):
        self.kg_csv_path = kg_path
        self.emb_dir = emb_dir
        self.output_dir = output_dir
        self.model_type = self._get_model_type(model_type)
        self.model = None
        self.entity_to_id = None
        self.relation_to_id = None

        # Grid search for embedding dimensions
        self.embedding_dims = embedding_dims if embedding_dims else [128, 192, 256]
        self.best_embedding_dim = None

        # Create directories
        os.makedirs(emb_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # Setup model_output files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.kge_results_file = os.path.join(output_dir, f"kge_training_results_{timestamp}.json")
        self.kge_metrics_file = os.path.join(output_dir, f"kge_training_metrics_{timestamp}.txt")

    def _get_model_type(self, model_type: Optional[str]) -> str:
        """Get and validate model type, with user input if needed"""
        if model_type and model_type.lower() in self.AVAILABLE_MODELS:
            return model_type.lower()

        print("=" * 60)
        print("PHASE 1: KGE MODEL SELECTION")
        print("=" * 60)
        print("Available KGE models:")
        for key, name in self.AVAILABLE_MODELS.items():
            print(f"  {key}: {name}")

        while True:
            user_input = input("\nSelect KGE model (default: transe): ").strip().lower()
            if not user_input:
                return 'transe'
            if user_input in self.AVAILABLE_MODELS:
                return user_input
            print(f"Invalid choice. Please select from: {list(self.AVAILABLE_MODELS.keys())}")

    def prepare_triples_factory(self) -> Tuple[TriplesFactory, TriplesFactory]:
        """Prepare triples factory for KGE training"""
        print("\nPreparing triples factory...")
        df = pd.read_csv(self.kg_csv_path)
        triples = df[['subject', 'relation', 'object']].values

        # Create triples factory
        tf = TriplesFactory.from_labeled_triples(
            triples,
            create_inverse_triples=False,
        )

        # Train/test split for KGE evaluation
        training_tf, testing_tf = tf.split([0.8, 0.2], random_state=0)

        print(f"Total triples: {len(triples)}")
        print(f"Training triples: {len(training_tf.triples)}")
        print(f"Testing triples: {len(testing_tf.triples)}")
        print(f"Entities: {training_tf.num_entities}")
        print(f"Relations: {training_tf.num_relations}")

        return training_tf, testing_tf

    def _get_model_kwargs(self, embedding_dim: int) -> Dict[str, Any]:
        """Get model-specific kwargs for PyKEEN"""
        base_kwargs = {
            'embedding_dim': embedding_dim,
        }

        if self.model_type == 'transe':
            base_kwargs.update({
                'scoring_fct_norm': 2,
                'regularizer': 'lp',
                'regularizer_kwargs': dict(p=3, weight=1e-5),
            })
        elif self.model_type == 'convkb':
            base_kwargs.update({
                'num_filters': 32,
                'hidden_dropout_rate': 0.5,
            })
        elif self.model_type == 'complex':
            base_kwargs.update({
                'regularizer': 'lp',
                'regularizer_kwargs': dict(weight=1e-5),
            })
        elif self.model_type == 'distmult':
            base_kwargs.update({
                'regularizer': 'lp',
                'regularizer_kwargs': dict(weight=1e-5),
            })
        elif self.model_type == 'rotate':
            base_kwargs.update({
                'regularizer': 'lp',
                'regularizer_kwargs': dict(weight=1e-5),
            })

        return base_kwargs

    def train_kge_model_single_dim(self, training_tf: TriplesFactory, testing_tf: TriplesFactory,
                                   embedding_dim: int) -> Tuple[Any, Dict[str, Any]]:
        """Train the KGE model with a specific embedding dimension"""
        print(f"\nTraining {self.AVAILABLE_MODELS[self.model_type]} model with embedding_dim={embedding_dim}...")
        print("This may take several minutes...")

        # Get model-specific parameters
        model_kwargs = self._get_model_kwargs(embedding_dim)
        negative_sampler = None
        negative_sampler_kwargs = None

        if self.model_type.lower() == 'complex':
            negative_sampler = 'bernoulli'
            negative_sampler_kwargs = dict(
                num_negs_per_pos=128            )

        # Train model with PyKEEN
        device =    "mps" if torch.backends.mps.is_available() else "cpu"
        stopper_kwargs = {
            'frequency': 5,  # Check every 5 epochs
            'patience': 3,  # Stop after 10 evaluations without improvement
            'relative_delta': 0.001,  # 0.1% minimum improvement
            'metric': 'mean_reciprocal_rank',  # Use MRR for early stopping
        }

        result = pipeline(
            training=training_tf,
            testing=testing_tf,
            model=self.model_type,
            device= device,
            model_kwargs=model_kwargs,
            negative_sampler=negative_sampler,
            negative_sampler_kwargs=negative_sampler_kwargs,
            optimizer='adam',
            optimizer_kwargs=dict(lr=1e-3),
            loss='negativeloglikelihood',
            training_kwargs=dict(
                num_epochs=200,
                batch_size=100,
                use_tqdm_batch=True,
            ),
            stopper='early',
            stopper_kwargs=stopper_kwargs,
            evaluator_kwargs=dict(filtered=True),
            random_seed=0,
        )

        # Extract evaluation metrics
        kge_metrics = {
            'model_type': self.model_type,
            'model_name': self.AVAILABLE_MODELS[self.model_type],
            'embedding_dim': embedding_dim,
            'num_entities': training_tf.num_entities,
            'num_relations': training_tf.num_relations,
            'training_triples': len(training_tf.triples),
            'testing_triples': len(testing_tf.triples),
        }

        # Add evaluation results if available
        if hasattr(result, 'metric_results') and result.metric_results:
            kge_metrics['evaluation'] = {
                'hits_at_1': float(result.metric_results.get_metric('hits@1')),
                'hits_at_3': float(result.metric_results.get_metric('hits@3')),
                'hits_at_10': float(result.metric_results.get_metric('hits@10')),
                'mrr': float(result.metric_results.get_metric('mean_reciprocal_rank')),
                'mr': float(result.metric_results.get_metric('mean_rank')),
            }

        print(f"\nKGE model training completed for embedding_dim={embedding_dim}!")
        if 'evaluation' in kge_metrics:
            print(f"Hits@1: {kge_metrics['evaluation']['hits_at_1']:.4f}")
            print(f"Hits@3: {kge_metrics['evaluation']['hits_at_3']:.4f}")
            print(f"Hits@10: {kge_metrics['evaluation']['hits_at_10']:.4f}")
            print(f"MRR: {kge_metrics['evaluation']['mrr']:.4f}")

        return result.model, kge_metrics

    def train_kge_model(self, training_tf: TriplesFactory, testing_tf: TriplesFactory) -> Dict[str, Any]:
        """Train the KGE model with grid search over embedding dimensions"""
        print("\n" + "=" * 60)
        print("GRID SEARCH: Testing different embedding dimensions")
        print(f"Embedding dimensions to test: {self.embedding_dims}")
        print("=" * 60)

        best_score = -float('inf')
        best_model = None
        best_metrics = None
        all_results = []

        for dim in self.embedding_dims:
            model, metrics = self.train_kge_model_single_dim(training_tf, testing_tf, dim)

            # Use MRR as the selection criterion (you can change this to hits@10 or other metrics)
            if 'evaluation' in metrics:
                score = metrics['evaluation']['mrr']
            else:
                score = 0.0

            metrics['grid_search_score'] = score
            all_results.append(metrics)

            print(f"\nEmbedding dim {dim}: Score (MRR) = {score:.4f}")

            if score > best_score:
                best_score = score
                best_model = model
                best_metrics = metrics
                self.best_embedding_dim = dim

        # Store the best model and mappings
        self.model = best_model
        self.entity_to_id = training_tf.entity_to_id
        self.relation_to_id = training_tf.relation_to_id

        # Add grid search results to best metrics
        best_metrics['grid_search_results'] = all_results
        best_metrics['best_embedding_dim'] = self.best_embedding_dim
        best_metrics['best_score'] = best_score

        print("\n" + "=" * 60)
        print("GRID SEARCH COMPLETED")
        print(f"Best embedding dimension: {self.best_embedding_dim}")
        print(f"Best score (MRR): {best_score:.4f}")
        print("=" * 60)

        return best_metrics

    def save_kge_model_and_mappings(self, kge_metrics: Dict[str, Any]) -> str:
        """Save the trained KGE model and entity/relation mappings"""
        # Save model
        model_path = os.path.join(self.emb_dir, f'{self.model_type}_model.pkl')
        torch.save(self.model, model_path)

        # Save entity and relation mappings
        mappings_path = os.path.join(self.emb_dir, f'{self.model_type}_mappings.pkl')
        mappings = {
            'entity_to_id': self.entity_to_id,
            'relation_to_id': self.relation_to_id,
            'model_type': self.model_type,
            'embedding_dim': self.best_embedding_dim
        }
        joblib.dump(mappings, mappings_path)

        # SOLO guardar información básica en JSON
        basic_info = {
            'timestamp': datetime.now().isoformat(),
            'model_path': model_path,
            'mappings_path': mappings_path,
            'model_type': self.model_type,
            'best_embedding_dim': self.best_embedding_dim,
            'best_score': float(kge_metrics.get('best_score', 0.0))
        }

        with open(self.kge_results_file, 'w') as f:
            json.dump(basic_info, f, indent=2, default=str)

        # Save human-readable metrics
        with open(self.kge_metrics_file, 'w') as f:
            f.write("KGE MODEL TRAINING RESULTS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Timestamp: {basic_info['timestamp']}\n")
            f.write(f"Model: {kge_metrics['model_name']} ({kge_metrics['model_type']})\n")
            f.write(f"Embedding Dimension: {kge_metrics['embedding_dim']}\n")
            f.write(f"Entities: {kge_metrics['num_entities']}\n")
            f.write(f"Relations: {kge_metrics['num_relations']}\n")
            f.write(f"Training Triples: {kge_metrics['training_triples']}\n")
            f.write(f"Testing Triples: {kge_metrics['testing_triples']}\n\n")

            if 'evaluation' in kge_metrics:
                f.write("EVALUATION METRICS\n")
                f.write("-" * 30 + "\n")
                f.write(f"Hits@1: {kge_metrics['evaluation']['hits_at_1']:.4f}\n")
                f.write(f"Hits@3: {kge_metrics['evaluation']['hits_at_3']:.4f}\n")
                f.write(f"Hits@10: {kge_metrics['evaluation']['hits_at_10']:.4f}\n")
                f.write(f"Mean Reciprocal Rank: {kge_metrics['evaluation']['mrr']:.4f}\n")
                f.write(f"Mean Rank: {kge_metrics['evaluation']['mr']:.4f}\n")

        print(f"\nKGE model and results saved:")
        print(f"  Model: {model_path}")
        print(f"  Mappings: {mappings_path}")
        print(f"  Results: {self.kge_results_file}")
        print(f"  Metrics: {self.kge_metrics_file}")

        return model_path


    def run_kge_training(self) -> str:
        """Run complete KGE training pipeline"""
        print("=" * 60)
        print("STARTING PHASE 1: KGE MODEL TRAINING")
        print("=" * 60)

        # Prepare data
        training_tf, testing_tf = self.prepare_triples_factory()

        # Train model with grid search
        kge_metrics = self.train_kge_model(training_tf, testing_tf)

        # Save model and mappings
        model_path = self.save_kge_model_and_mappings(kge_metrics)

        print("\n" + "=" * 60)
        print("PHASE 1 COMPLETED: KGE MODEL TRAINING FINISHED")
        print("=" * 60)

        return model_path


class TopicClassifier:
    """
    Phase 2: Topic classification using KGE embeddings
    """

    def __init__(self, kge_csv_path: str, model_path: str, mappings_path: str,
                 output_dir: str = "data/model_output/amazon"):
        self.kge_csv_path = kge_csv_path
        self.model_path = model_path
        self.mappings_path = mappings_path
        self.output_dir = output_dir

        self.kge_model = None
        self.entity_to_id = None
        self.relation_to_id = None
        self.model_type = None
        self.embedding_dim = None  # STORE THE EMBEDDING DIMENSION
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
        print("\nLoading KGE model and mappings...")
        self.kge_model = torch.load(self.model_path, weights_only=False)
        mappings = joblib.load(self.mappings_path)

        self.entity_to_id = mappings['entity_to_id']
        self.relation_to_id = mappings['relation_to_id']
        self.model_type = mappings['model_type']

        # LOAD THE EMBEDDING DIMENSION FROM MAPPINGS
        self.embedding_dim = mappings.get('embedding_dim', 150)  # Default to 150 if not found

        print(f"Model type: {self.model_type}")
        print(f"Embedding dimension: {self.embedding_dim}")
        print(f"Entities: {len(self.entity_to_id)}")
        print(f"Relations: {len(self.relation_to_id)}")

    def load_classification_data(self) -> pd.DataFrame:
        """Load and prepare classification data"""
        print("\nLoading classification data...")
        df = pd.read_csv(self.kge_csv_path)

        # Determine if binary or multiclass
        unique_topics = df['topic'].nunique()
        self.is_binary_classification = unique_topics == 2

        print(f"Total samples: {len(df)}")
        print(f"Unique topics: {unique_topics}")
        print(f"Classification type: {'Binary' if self.is_binary_classification else 'Multiclass'}")
        print("\nTopic distribution:")
        print(df['topic'].value_counts())

        return df

    def get_entity_embedding(self, entity: str) -> np.ndarray:
        """Get embedding for an entity"""
        if entity not in self.entity_to_id:
            # Return zero vector for unknown entities with correct dimension
            return np.zeros(self.embedding_dim)

        entity_id = self.entity_to_id[entity]
        entity_tensor = torch.tensor([entity_id], dtype=torch.long)

        with torch.no_grad():
            embedding = self.kge_model.entity_representations[0](entity_tensor)

        return embedding.cpu().numpy().flatten()

    def create_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create features from KGE embeddings"""
        print("\nCreating features from KGE embeddings...")

        X_list = []
        y_list = []

        for idx, row in df.iterrows():
            subject_emb = self.get_entity_embedding(row['subject'])
            object_emb = self.get_entity_embedding(row['object'])

            # Concatenate embeddings
            combined_emb = np.concatenate([subject_emb, object_emb])
            X_list.append(combined_emb)
            y_list.append(row['topic'])

            if (idx + 1) % 1000 == 0:
                print(f"Processed {idx + 1}/{len(df)} samples...")

        X = np.array(X_list)
        y = np.array(y_list)

        print(f"\nFeature matrix shape: {X.shape}")
        print(f"Target vector shape: {y.shape}")
        print(f"Feature dimension per entity: {self.embedding_dim}")
        print(f"Total feature dimension: {self.embedding_dim * 2}")

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
                    # One-vs-Rest AUC
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
            'embedding_dim': self.embedding_dim,  # SAVE THE EMBEDDING DIM USED
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

        # Load KGE model and mappings
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


# Main execution functions for each phase
def run_kge_training_phase(kg_path: str, emb_dir: str,
                           output_dir: str = "data/model_output/amazon", model_type: str = None,
                           embedding_dims: List[int] = None) -> str:
    """
    Execute Phase 1: KGE Model Training

    Args:
        kg_path: Path to knowledge graph TSV file
        emb_dir: Directory for embeddings storage
        output_dir: Directory for model_output files
        model_type: KGE model type (transe, convkb, etc.)
        embedding_dims: List of embedding dimensions for grid search (default: [128, 192, 256])

    Returns:
        Path to the trained model
    """
    trainer = KGEModelTrainer(kg_path=kg_path, emb_dir=emb_dir, output_dir=output_dir,
                              model_type=model_type, embedding_dims=embedding_dims)

    return trainer.run_kge_training()


def run_classification_phase(kg_path: str, model_path: str, mappings_path: str,
                             output_dir: str = "data/model_output/amazon",
                             use_random_search: bool = True, n_iter: int = 20, cv: int = 3) -> Tuple[
    XGBClassifier, Dict[str, Any]]:
    """
    Execute Phase 2: Topic Classification

    Args:
        kg_path: Path to csv triples files
        model_path: Path to trained KGE model
        mappings_path: Path to entity/relation mappings
        output_dir: Directory for model_output files
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


def run_complete_pipeline(kg_path: str, emb_dir: str,
                          output_dir: str = "data/model_output/amazon", model_type: str = None,
                          embedding_dims: List[int] = None,
                          use_random_search: bool = True, n_iter: int = 20, cv: int = 3) -> Tuple[
    XGBClassifier, Dict[str, Any]]:
    """
    Execute both phases sequentially

    Args:
        kg_path: Path to knowledge graph TSV file
        emb_dir: Directory for embeddings storage
        output_dir: Directory for model_output files
        model_type: KGE model type (transe, convkb, etc.)
        embedding_dims: List of embedding dimensions for grid search (default: [128, 192, 256])
        use_random_search: Use RandomizedSearchCV if True, GridSearchCV if False
        n_iter: Number of iterations for RandomizedSearchCV
        cv: Number of cross-validation folds

    Returns:
        Tuple of (best_model, metrics)
    """
    print("STARTING COMPLETE TWO-PHASE PIPELINE")
    print("=" * 60)

    # Phase 1: KGE Training
    model_path = run_kge_training_phase(
        kg_path=kg_path,
        emb_dir=emb_dir,
        output_dir=output_dir,
        model_type=model_type,
        embedding_dims=embedding_dims
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

    print("\n" + "=" * 60)
    print("COMPLETE PIPELINE FINISHED SUCCESSFULLY!")
    print("=" * 60)

    return best_model, metrics


# Example usage functions for demonstration
def example_phase_1():
    """Example of running Phase 1 only"""
    model_path = run_kge_training_phase(
        kg_path="../olds/data/triples_raw/amazon/dataset_triplet_reuters_activities_new_simplificado.csv",
        emb_dir="../olds/data/triples_emb/amazon",
        output_dir="../olds/data/model_output/amazon",
        model_type="transe",  # or None for interactive selection
        embedding_dims=[128, 192, 256]
    )
    print(f"Phase 1 completed. Model saved to: {model_path}")
    return model_path


def example_phase_2(model_path: str, mappings_path: str):
    """Example of running Phase 2 only"""
    best_model, metrics = run_classification_phase(
        kg_path="../olds/data/triples_raw/amazon/dataset_triplet_reuters_activities_new_simplificado.csv",
        model_path=model_path,
        mappings_path=mappings_path,
        output_dir="../olds/data/model_output/repo",
        use_random_search=True,
        n_iter=30,
        cv=5
    )
    print(f"Phase 2 completed. Best accuracy: {metrics['accuracy']:.4f}")
    return best_model, metrics


def example_complete_pipeline():
    """Example of running both phases sequentially"""
    best_model, metrics = run_complete_pipeline(
        kg_path="../data/triples_raw/amazonocion2/dataset_triplet_reuters_activities_new_simplificado.csv",
        emb_dir="../data/triples_emb/amazon/",
        output_dir="../data/model_output/amazon",
        model_type=None,
        embedding_dims=[128, 192, 256],
        use_random_search=True,
        n_iter=25,
        cv=3
    )
    print(f"Complete pipeline finished. Final accuracy: {metrics['accuracy']:.4f}")
    return best_model, metrics


def interactive_menu():
    """Interactive menu for selecting operations"""
    print("\n" + "=" * 60)
    print("KGE CLASSIFICATION MODULE - INTERACTIVE MENU")
    print("=" * 60)

    # Repository selection
    print("\nAvailable repositories:")
    print("1. Amazon")
    print("2. BBC")
    print("3. arXiv")
    print("=" * 60)

    repo_choice = input("\nSelect repository (1-3): ").strip()

    if repo_choice == '1':
        repo = 'amazon'
        dataset_name = 'dataset_triplet_reuters_activities_new_simplificado.csv'
    elif repo_choice == '2':
        repo = 'bbc'
        dataset_name = 'dataset_triplet_bbc_new_simplificado.csv'
    elif repo_choice == '3':
        repo = 'arxiv'
        dataset_name = 'dataset_triplet_arxiv_new_simplificado.csv'
    else:
        print("Invalid repository choice. Exiting...")
        sys.exit(1)

    print(f"\nSelected repository: {repo.upper()}")

    print("\nSelect an operation:")
    print("1. Phase 1: KGE Training Only")
    print("2. Phase 2: Classification Only (requires trained KGE model)")
    print("3. Complete Pipeline (Phase 1 + Phase 2)")
    print("4. Exit")
    print("=" * 60)

    choice = input("\nEnter your choice (1-4): ").strip()

    if choice == '4':
        print("Exiting...")
        sys.exit(0)

    # Define paths based on repository
    base_path = "../data"
    kg_path = f"{base_path}/triples_raw/{repo}/{dataset_name}"
    emb_dir = f"{base_path}/triples_emb/{repo}_2iter"
    model_path = f"{emb_dir}/transe_model.pkl"
    mappings_path = f"{emb_dir}/transe_mappings.pkl"

    # Output directory with _2iter suffix
    output_dir = f"{base_path}/model_output/{repo}_2iter"

    if choice == '1':
        # Phase 1: KGE Training
        print("\n--- Phase 1: KGE Training ---")
        model_type = input(
            "Enter model type (transe/convkb/complex/distmult/rotate) or press Enter for interactive selection: ").strip().lower() or None
        embedding_dims_input = input(
            "Enter embedding dimensions to test (comma-separated, e.g., 128,192,256) or press Enter for default [128,192,256]: ").strip()

        if embedding_dims_input:
            embedding_dims = [int(x.strip()) for x in embedding_dims_input.split(',')]
        else:
            embedding_dims = [128, 192, 256]

        model_path = run_kge_training_phase(
            kg_path=kg_path,
            emb_dir=emb_dir,
            output_dir=output_dir,
            model_type=model_type,
            embedding_dims=embedding_dims
        )
        print(f"\nPhase 1 completed. Model saved to: {model_path}")

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
        model_type = input(
            "Enter model type (transe/convkb/complex/distmult/rotate) or press Enter for interactive selection: ").strip().lower() or None
        embedding_dims_input = input(
            "Enter embedding dimensions to test (comma-separated, e.g., 128,192,256) or press Enter for default [128,192,256]: ").strip()

        if embedding_dims_input:
            embedding_dims = [int(x.strip()) for x in embedding_dims_input.split(',')]
        else:
            embedding_dims = [128, 192, 256]

        use_random_search_input = input("Use RandomizedSearchCV? (y/n, default: y): ").strip().lower()
        use_random_search = use_random_search_input != 'n'

        if use_random_search:
            n_iter = int(input("Number of iterations for RandomizedSearchCV (default: 20): ").strip() or "20")
        else:
            n_iter = 20

        cv = int(input("Number of cross-validation folds (default: 3): ").strip() or "3")

        best_model, metrics = run_complete_pipeline(
            kg_path=kg_path,
            emb_dir=emb_dir,
            output_dir=output_dir,
            model_type=model_type,
            embedding_dims=embedding_dims,
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
        # Programmatic mode - parse arguments
        print("KGE Classification Module - Programmatic Mode")
        print("=" * 60)

        # Example: python script.py phase repo_name
        if len(sys.argv) >= 3:
            phase = sys.argv[1]
            repo = sys.argv[2].lower()

            # Validate repository
            if repo not in ['amazon', 'bbc']:
                print(f"Error: Invalid repository '{repo}'")
                print("Available repositories: amazon, bbc")
                sys.exit(1)

            # Set dataset name based on repository
            if repo == 'amazon':
                dataset_name = 'dataset_triplet_reuters_activities_new_simplificado.csv'
            elif repo == 'bbc':
                dataset_name = 'dataset_triplet_bbc_new_simplificado.csv'

            base_path = "../olds/data"
            kg_path = f"{base_path}/triples_raw/{repo}/{dataset_name}"
            emb_dir = f"{base_path}/triples_emb/{repo}"
            model_path = f"{emb_dir}/transe_model.pkl"
            mappings_path = f"{emb_dir}/transe_mappings.pkl"

            # Output directory with _2iter suffix
            output_dir = f"{base_path}/model_output/{repo}_2iter"

            print(f"Repository: {repo.upper()}")
            print(f"Dataset: {dataset_name}")
            print(f"Output directory: {output_dir}")
            print("=" * 60)

            if phase == "phase1":
                run_kge_training_phase(
                    kg_path=kg_path,
                    emb_dir=emb_dir,
                    output_dir=output_dir,
                    embedding_dims=[128, 192, 256]
                )
            elif phase == "phase2":
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
                    embedding_dims=[128, 192, 256]
                )
            else:
                print(f"Unknown phase: {phase}")
                print("Available phases: phase1, phase2, complete")
        else:
            print("Usage: python script.py <phase> <repo_name>")
            print("Phases: phase1, phase2, complete")
            print("Repositories: amazon, bbc")
    else:
        # Interactive mode
        print("KGE Classification Module - Interactive Mode")
        print("Available functions:")
        print("- run_kge_training_phase(): Execute Phase 1 (KGE training)")
        print("- run_classification_phase(): Execute Phase 2 (Topic classification)")
        print("- run_complete_pipeline(): Execute both phases sequentially")
        print("\nFor examples, see example_phase_1(), example_phase_2(), example_complete_pipeline()")

        interactive_menu()
