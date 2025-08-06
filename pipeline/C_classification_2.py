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
                 output_dir: str = "data/model_output/amazon", model_type: str = None):
        self.kg_csv_path = kg_path
        self.emb_dir = emb_dir
        self.output_dir = output_dir
        self.model_type = self._get_model_type(model_type)
        self.model = None
        self.entity_to_id = None
        self.relation_to_id = None

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

    def _get_model_kwargs(self) -> Dict[str, Any]:
        """Get model-specific kwargs for PyKEEN"""
        base_kwargs = {
            'embedding_dim': 150,
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

    def train_kge_model(self, training_tf: TriplesFactory, testing_tf: TriplesFactory) -> Dict[str, Any]:
        """Train the KGE model"""
        print(f"\nTraining {self.AVAILABLE_MODELS[self.model_type]} model...")
        print("This may take several minutes...")

        # Get model-specific parameters
        model_kwargs = self._get_model_kwargs()

        # Train model with PyKEEN
        result = pipeline(
            training=training_tf,
            testing=testing_tf,
            model=self.model_type,
            model_kwargs=model_kwargs,
            optimizer='adam',
            optimizer_kwargs=dict(lr=1e-3),
            loss='negativeloglikelihood',
            training_kwargs=dict(
                num_epochs=200,
                batch_size=100,
                use_tqdm_batch=True,
            ),
            evaluator_kwargs=dict(filtered=True),
            random_seed=0,
        )

        self.model = result.model
        self.entity_to_id = training_tf.entity_to_id
        self.relation_to_id = training_tf.relation_to_id

        # Extract evaluation metrics
        kge_metrics = {
            'model_type': self.model_type,
            'model_name': self.AVAILABLE_MODELS[self.model_type],
            'embedding_dim': model_kwargs['embedding_dim'],
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

        print(f"\nKGE model training completed!")
        if 'evaluation' in kge_metrics:
            print(f"Hits@1: {kge_metrics['evaluation']['hits_at_1']:.4f}")
            print(f"Hits@3: {kge_metrics['evaluation']['hits_at_3']:.4f}")
            print(f"Hits@10: {kge_metrics['evaluation']['hits_at_10']:.4f}")
            print(f"MRR: {kge_metrics['evaluation']['mrr']:.4f}")

        return kge_metrics

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
            'model_type': self.model_type
        }
        joblib.dump(mappings, mappings_path)

        # Save KGE training results
        kge_results = {
            'timestamp': datetime.now().isoformat(),
            'model_path': model_path,
            'mappings_path': mappings_path,
            'metrics': kge_metrics
        }

        with open(self.kge_results_file, 'w') as f:
            json.dump(kge_results, f, indent=2, default=str)

        # Save human-readable metrics
        with open(self.kge_metrics_file, 'w') as f:
            f.write("KGE MODEL TRAINING RESULTS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Timestamp: {kge_results['timestamp']}\n")
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
        """Run complete KGE training phase"""
        print("=" * 60)
        print("STARTING PHASE 1: KGE MODEL TRAINING")
        print("=" * 60)

        # Load data
        #df = self.load_data()

        # Prepare triples
        training_tf, testing_tf = self.prepare_triples_factory()

        # Train model
        kge_metrics = self.train_kge_model(training_tf, testing_tf)

        # Save model and mappings
        model_path = self.save_kge_model_and_mappings(kge_metrics)

        print("\n" + "=" * 60)
        print("PHASE 1 COMPLETED: KGE MODEL READY")
        print("=" * 60)
        print(f"Model saved to: {model_path}")
        print("Ready for Phase 2: Classification")

        return model_path


class TopicClassifier:
    """
    Phase 2: Topic classification using KGE embeddings
    """

    def __init__(self, kge_csv_path: str, model_path:str, mappings_path: str,
                 output_dir: str = "data/model_output/amazon"):
        self.kge_csv_path = kge_csv_path
        self.mappings_path = mappings_path
        self.model_path = model_path
        self.output_dir = output_dir
        self.model = None
        self.entity_to_id = None
        self.relation_to_id = None
        self.entity_embeddings = None
        self.relation_embeddings = None
        self.is_binary_classification = None
        self.model_type = None

        # Setup model_output files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.classification_results_file = os.path.join(output_dir, f"topic_classification_results_{timestamp}.json")
        self.classification_metrics_file = os.path.join(output_dir, f"topic_classification_metrics_{timestamp}.txt")
        self.best_model_file = os.path.join(output_dir, f"best_topic_classifier_{timestamp}.pkl")

    def load_kge_model_and_mappings(self):
        """Load the trained KGE model and mappings"""
        print("Loading trained KGE model and mappings...")

        # Load model
        self.model = torch.load(self.model_path, map_location='cpu', weights_only=False)
        self.model.eval()

        # Load mappings
        mappings = joblib.load(self.mappings_path)
        self.entity_to_id = mappings['entity_to_id']
        self.relation_to_id = mappings['relation_to_id']
        self.model_type = mappings['model_type']

        # Extract embeddings
        self.entity_embeddings = self.model.entity_representations[0](indices=None)
        self.entity_embeddings = self.entity_embeddings.detach().cpu().numpy()

        self.relation_embeddings = self.model.relation_representations[0](indices=None)
        self.relation_embeddings = self.relation_embeddings.detach().cpu().numpy()

        print(f"Loaded {self.model_type} model")
        print(f"Entity embeddings shape: {self.entity_embeddings.shape}")
        print(f"Relation embeddings shape: {self.relation_embeddings.shape}")

    def load_classification_data(self) -> pd.DataFrame:
        """Load data for topic classification"""
        print("\nLoading dataset for topic classification...")
        df = pd.read_csv(self.kge_csv_path)

        # Clean data
        df = df[df['subject'].notna() & df['object'].notna()]
        df['relation'] = df['relation'].fillna('have')

        # Determine classification type
        self.is_binary_classification = len(df['new_topic'].unique()) == 2

        print(f"Dataset shape: {df.shape}")
        print(f"Classification type: {'Binary' if self.is_binary_classification else 'Multiclass'}")
        print(f"Classes: {sorted(df['new_topic'].unique())}")
        print(f"Class distribution: {df['new_topic'].value_counts().to_dict()}")

        return df

    def get_embedding(self, entity: str) -> np.ndarray:
        """Get embedding for a specific entity"""
        idx = self.entity_to_id.get(entity)
        if idx is not None:
            return self.entity_embeddings[idx]
        else:
            return np.zeros(self.entity_embeddings.shape[1])

    def create_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create feature matrix from embeddings"""
        print("\nCreating feature matrix from embeddings...")

        # Get embeddings for subjects and objects
        subject_embs = np.vstack(df['subject'].apply(self.get_embedding).values)
        object_embs = np.vstack(df['object'].apply(self.get_embedding).values)

        # Concatenate subject and object embeddings
        X = np.hstack([subject_embs, object_embs])
        y = df['new_topic'].astype(int).values

        print(f"Feature matrix shape: {X.shape}")
        print(f"Target distribution: {np.bincount(y)}")

        return X, y

    def get_grid_search_params(self) -> Dict[str, Any]:
        """Define grid search parameters for XGBoost"""
        param_grid = {
            'n_estimators': [150, 250, 450],
            'learning_rate': [0.01, 0.1, 0.3],
            'max_depth': [3, 6, 10],
            'subsample': [0.4, 0.6, 0.8],
            'colsample_bytree': [0.4, 0.6, 0.8],
            'reg_alpha': [0, 0.5, 1],
            'reg_lambda': [0, 0.5, 1],
        }

        return param_grid

    def calculate_mean_per_class_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate mean per class error"""
        cm = confusion_matrix(y_true, y_pred)
        per_class_errors = []

        for i in range(len(cm)):
            if cm[i].sum() > 0:
                class_error = 1 - (cm[i, i] / cm[i].sum())
                per_class_errors.append(class_error)

        return np.mean(per_class_errors)

    def calculate_auc_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calculate AUC metrics"""
        auc_metrics = {}

        if self.is_binary_classification:
            # Binary classification AUC
            if y_pred_proba.shape[1] == 2:
                y_scores = y_pred_proba[:, 1]
            else:
                y_scores = y_pred_proba.ravel()

            auc_score = roc_auc_score(y_true, y_scores)
            auc_metrics['roc_auc'] = auc_score

            # Get ROC curve data
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            auc_metrics['fpr'] = fpr.tolist()
            auc_metrics['tpr'] = tpr.tolist()
            auc_metrics['thresholds'] = thresholds.tolist()

        else:
            # Multiclass AUC (one-vs-rest)
            try:
                auc_score = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
                auc_metrics['roc_auc_macro'] = auc_score

                # Per-class AUC
                n_classes = y_pred_proba.shape[1]
                class_aucs = []
                for i in range(n_classes):
                    if i in y_true:
                        y_true_binary = (y_true == i).astype(int)
                        y_scores_binary = y_pred_proba[:, i]
                        class_auc = roc_auc_score(y_true_binary, y_scores_binary)
                        class_aucs.append(class_auc)
                    else:
                        class_aucs.append(None)

                auc_metrics['per_class_auc'] = class_aucs

            except ValueError as e:
                print(f"Could not calculate multiclass AUC: {e}")
                auc_metrics['roc_auc_macro'] = None

        return auc_metrics

    def evaluate_model(self, model: XGBClassifier, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Comprehensive model evaluation"""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        logloss = log_loss(y_test, y_pred_proba)
        mean_per_class_error = self.calculate_mean_per_class_error(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        # AUC metrics
        auc_metrics = self.calculate_auc_metrics(y_test, y_pred_proba)

        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)

        metrics = {
            'accuracy': accuracy,
            'logloss': logloss,
            'mean_per_class_error': mean_per_class_error,
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'auc_metrics': auc_metrics,
            'is_binary_classification': self.is_binary_classification,
            'model_type': self.model_type,
            'n_classes': len(np.unique(y_test))
        }

        return metrics

    def save_classification_results(self, metrics: Dict[str, Any], best_params: Dict[str, Any],
                                    cv_results: List[Dict[str, Any]]) -> None:
        """Save classification results to files"""

        # Prepare results dictionary
        results = {
            'timestamp': datetime.now().isoformat(),
            'model_info': {
                'kge_model': self.model_type,
                'classifier': 'XGBoost',
                'is_binary_classification': self.is_binary_classification
            },
            'best_parameters': best_params,
            'test_metrics': {
                'accuracy': metrics['accuracy'],
                'logloss': metrics['logloss'],
                'mean_per_class_error': metrics['mean_per_class_error'],
                'n_classes': metrics['n_classes']
            },
            'auc_metrics': metrics['auc_metrics'],
            'confusion_matrix': metrics['confusion_matrix'],
            'classification_report': metrics['classification_report'],
            'top_cv_results': cv_results
        }

        # Save JSON results
        with open(self.classification_results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Save human-readable metrics
        with open(self.classification_metrics_file, 'w') as f:
            f.write("TOPIC CLASSIFICATION RESULTS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Timestamp: {results['timestamp']}\n")
            f.write(f"KGE Model: {self.model_type}\n")
            f.write(f"Classification Type: {'Binary' if self.is_binary_classification else 'Multiclass'}\n")
            f.write(f"Number of Classes: {metrics['n_classes']}\n\n")

            f.write("PERFORMANCE METRICS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Log Loss: {metrics['logloss']:.4f}\n")
            f.write(f"Mean Per-Class Error: {metrics['mean_per_class_error']:.4f}\n")

            if 'roc_auc' in metrics['auc_metrics']:
                f.write(f"ROC AUC: {metrics['auc_metrics']['roc_auc']:.4f}\n")
            elif 'roc_auc_macro' in metrics['auc_metrics'] and metrics['auc_metrics']['roc_auc_macro']:
                f.write(f"ROC AUC (Macro): {metrics['auc_metrics']['roc_auc_macro']:.4f}\n")

            f.write(f"\nConfusion Matrix:\n{np.array(metrics['confusion_matrix'])}\n\n")

            f.write("BEST PARAMETERS\n")
            f.write("-" * 30 + "\n")
            for param, value in best_params.items():
                f.write(f"{param}: {value}\n")

        print(f"\nClassification results saved to:")
        print(f"  JSON: {self.classification_results_file}")
        print(f"  Text: {self.classification_metrics_file}")
        print(f"  Model: {self.best_model_file}")

    def run_classification_with_grid_search(self, X: np.ndarray, y: np.ndarray,
                                            use_random_search: bool = True,
                                            n_iter: int = 20, cv: int = 3) -> Tuple[XGBClassifier, Dict[str, Any]]:
        """Run classification with grid search"""
        print(f"\nStarting topic classification with grid search...")
        print(f"Using {self.model_type} embeddings")

        # Train/test split for classification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0, stratify=y
        )

        # Base XGBoost classifier
        base_clf = XGBClassifier(
            random_state=0,
            use_label_encoder=False,
            eval_metric='mlogloss',
            tree_method='hist'
        )

        # Get parameter grid
        param_grid = self.get_grid_search_params()

        # Choose search strategy
        if use_random_search:
            print(f"Using RandomizedSearchCV with {n_iter} iterations...")
            search = RandomizedSearchCV(
                base_clf,
                param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring='neg_log_loss',
                random_state=0,
                n_jobs=-1,
                verbose=1
            )
        else:
            print("Using GridSearchCV...")
            search = GridSearchCV(
                base_clf,
                param_grid,
                cv=cv,
                scoring='neg_log_loss',
                n_jobs=-1,
                verbose=1
            )

        # Fit the search
        search.fit(X_train, y_train)

        # Get best model
        best_model = search.best_estimator_

        print(f"\nBest parameters found:")
        for param, value in search.best_params_.items():
            print(f"  {param}: {value}")

        print(f"Best CV score (neg_log_loss): {search.best_score_:.4f}")

        # Evaluate on test set
        test_metrics = self.evaluate_model(best_model, X_test, y_test)

        # Prepare CV results
        results_df = pd.DataFrame(search.cv_results_)
        top_models = results_df.nlargest(5, 'mean_test_score')[
            ['mean_test_score', 'std_test_score', 'params']
        ]

        cv_results = []
        for idx, row in top_models.iterrows():
            cv_results.append({
                'rank': len(cv_results) + 1,
                'mean_score': float(row['mean_test_score']),
                'std_score': float(row['std_test_score']),
                'params': dict(row['params'])
            })

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
                           output_dir: str = "data/model_output/amazon", model_type: str = None) -> str:
    """
    Execute Phase 1: KGE Model Training

    Args:
        kg_path: Path to knowledge graph TSV file
        emb_dir: Directory for embeddings storage
        output_dir: Directory for model_output files
        model_type: KGE model type (transe, convkb, etc.)

    Returns:
        Path to the trained model
    """
    trainer = KGEModelTrainer(kg_path=kg_path, emb_dir=emb_dir, output_dir=output_dir, model_type=model_type)

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
                          use_random_search: bool = True, n_iter: int = 20, cv: int = 3) -> Tuple[
    XGBClassifier, Dict[str, Any]]:
    """
    Execute both phases sequentially

    Args:
        kg_path: Path to knowledge graph TSV file
        emb_dir: Directory for embeddings storage
        output_dir: Directory for model_output files
        model_type: KGE model type (transe, convkb, etc.)
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
        model_type=model_type
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
        kg_path="data/triples_raw/amazon/dataset_triplet_amazon_new_simplificado.csv",
        emb_dir="data/triples_emb/amazon",
        output_dir="data/model_output/amazon",
        model_type="transe"  # or None for interactive selection
    )
    print(f"Phase 1 completed. Model saved to: {model_path}")
    return model_path


def example_phase_2(model_path: str, mappings_path: str):
    """Example of running Phase 2 only"""
    best_model, metrics = run_classification_phase(
        kg_path="data/triples_raw/amazon/dataset_triplet_amazon_new_simplificado.csv",
        model_path=model_path,
        mappings_path=mappings_path,
        output_dir="data/model_output/repo",
        use_random_search=True,
        n_iter=30,
        cv=5
    )
    print(f"Phase 2 completed. Best accuracy: {metrics['accuracy']:.4f}")
    return best_model, metrics


def example_complete_pipeline():
    """Example of running both phases sequentially"""
    best_model, metrics = run_complete_pipeline(
        kg_path="data/triples_raw/amazonocion2/dataset_triplet_amazon_new_simplificado.csv",
        emb_dir="data/triples_emb/amazon/",
        output_dir="data/model_output/amazon",
        model_type=None,
        use_random_search=True,
        n_iter=25,
        cv=3
    )
    print(f"Complete pipeline finished. Final accuracy: {metrics['accuracy']:.4f}")
    return best_model, metrics


if __name__ == "__main__":
    # This will only run if the script is executed directly, not when imported
    print("KGE Classification Module")
    print("Available functions:")
    print("- run_kge_training_phase(): Execute Phase 1 (KGE training)")
    print("- run_classification_phase(): Execute Phase 2 (Topic classification)")
    print("- run_complete_pipeline(): Execute both phases sequentially")
    print("\nFor examples, see example_phase_1(), example_phase_2(), example_complete_pipeline()")
    run_complete_pipeline(
        kg_path="data/triples_raw/amazon/dataset_triplet_amazon_new_simplificado.csv",
       #F model_path='data/triples_emb/amazon/kge_midek.pkl',
        emb_dir="data/triples_emb/amazon/",
        output_dir="data/model_output/amazon")
    #run_classification_phase( kg_path="data/triples_raw/amazon/dataset_triplet_amazon_new_simplificado.csv",model_path='data/triples_emb/amazon/transe_model.pkl', mappings_path='data/triples_emb/amazon/transe_mappings.pkl')
