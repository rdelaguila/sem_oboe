"""
Component B: Topic Modeling with LDA - oBOE Framework

Performs topic extraction, model selection, and classification pipeline:
- LDA-based topic modeling with hyperparameter optimization
- Coherence-based model evaluation
- Interactive optimal topic selection
- Text preprocessing with spaCy lemmatization
- Classifier training (SVM or XGBoost)

Requirements:
- gensim, scikit-learn, spacy, pandas, numpy
- Trained spaCy model: en_core_web_sm
"""
import multiprocessing as mp
import argparse


def main():
    import os
    import pickle
    import joblib
    import pandas as pd
    import numpy as np
    import spacy
    from gensim import corpora
    from gensim.models import LdaModel, CoherenceModel
    from sklearn.metrics import classification_report, confusion_matrix, log_loss
    from sklearn.metrics import roc_auc_score, accuracy_score
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV, cross_val_score
    from sklearn.preprocessing import LabelEncoder
    import warnings
    warnings.filterwarnings('ignore')

    # ---------- COMMAND-LINE ARGUMENTS ----------
    parser = argparse.ArgumentParser(description='LDA Topic Modeling with Classification')
    parser.add_argument('--classifier', type=str, choices=['svm', 'xgboost'],
                        default='xgboost', help='Choose classifier: svm or xgboost')
    parser.add_argument('--raw_path', type=str, default="data/processed/amazon/amazon_processed_semantic.pkl",
                        help='Path to raw data file')
    parser.add_argument('--eval_dir', type=str, default="data/lda_eval/amazon",
                        help='Directory to save LDA evaluation results')
    parser.add_argument('--max_topics', type=int, default=None,
                        help='Maximum number of topics to test (overrides default)')

    # Parse arguments with error handling
    try:
        args = parser.parse_args()
    except SystemExit:
        # If no arguments or error in arguments, use default values
        print("No valid arguments provided. Using default values:")
        args = argparse.Namespace(
            classifier='xgboost',
            raw_path="../olds/data/corpus_raw/amazon/corpus_raw",
            eval_dir="../olds/data/lda_eval/amazon",
            max_topics=None
        )

    # ---------- CONFIG ----------
    RAW_PATH = args.raw_path
    LDA_EVAL_DIR = args.eval_dir
    CLASSIFIER_TYPE = args.classifier
    MAX_TOPICS_ARG = args.max_topics

    # Display current configuration
    print("=" * 50)
    print("PIPELINE CONFIGURATION")
    print("=" * 50)
    print(f"Classifier: {CLASSIFIER_TYPE}")
    print(f"Data file: {RAW_PATH}")
    print(f"Output directory: {LDA_EVAL_DIR}")
    if MAX_TOPICS_ARG:
        print(f"Maximum topics (argument): {MAX_TOPICS_ARG}")
    print("=" * 50)

    # Verify data file exists
    if not os.path.exists(RAW_PATH):
        print(f"ERROR: Data file not found at: {RAW_PATH}")
        print("Please run preprocessing first or verify the path.")
        return

    os.makedirs(LDA_EVAL_DIR, exist_ok=True)

    # ---------- TOPIC PARAMETERS CONFIGURATION ----------
    print("\n" + "=" * 60)
    print("TOPIC MODELING PARAMETERS CONFIGURATION")
    print("=" * 60)

    # Alpha parameter
    ALPHAS = list(np.arange(0.009, 0.15, 0.06))
    ALPHAS.append('symmetric')
    ALPHAS.append('asymmetric')
    # Beta parameter
    BETAS = list(np.arange(0.5, 1, 0.04))
    BETAS.append('symmetric')

    # Configuration of maximum number of topics
    DEFAULT_MAX_TOPICS = 15
    START = 2
    STEP = 1

    # Calculate number of hyperparameter combinations
    num_alphas = len(ALPHAS)
    num_betas = len(BETAS)

    # If argument provided, use it
    if MAX_TOPICS_ARG:
        LIMIT = MAX_TOPICS_ARG
        print(f"Using maximum topics from arguments: {LIMIT}")
    else:
        # Request maximum topics from user
        print(f"Current config: evaluate from {START} to {DEFAULT_MAX_TOPICS} topics (step {STEP})")
        print("This will evaluate hyperparameter combinations for each number of topics.")

        # Calculate total combinations
        num_topic_values = (DEFAULT_MAX_TOPICS - START) // STEP + 1
        total_combinations = num_alphas * num_betas * num_topic_values

        print(f"Total combinations to evaluate: {total_combinations}")
        print(f"(Alpha: {num_alphas} × Beta: {num_betas} × Topics: {num_topic_values})")

        while True:
            try:
                user_input = input(
                    f"\nMaximum number of topics to evaluate? (Enter = {DEFAULT_MAX_TOPICS}, min=2): ").strip()

                if user_input == "":
                    LIMIT = DEFAULT_MAX_TOPICS
                    print(f"Using default value: {LIMIT} maximum topics")
                    break
                elif user_input.isdigit():
                    user_limit = int(user_input)
                    if user_limit < 2:
                        print("The minimum number of topics must be 2. Try again.")
                        continue
                    elif user_limit > 50:
                        confirm = input(
                            f"Confirm evaluation up to {user_limit} topics? This may take a long time (y/n): ").strip().lower()
                        if confirm in ['y', 'yes']:
                            LIMIT = user_limit
                            break
                        else:
                            continue
                    else:
                        LIMIT = user_limit
                        break
                else:
                    print("Please enter a number or press Enter.")

            except KeyboardInterrupt:
                print(f"\nUsing default value: {DEFAULT_MAX_TOPICS}")
                LIMIT = DEFAULT_MAX_TOPICS
                break
            except:
                print("Invalid input. Try again.")

    # Recalculate and display final configuration
    final_topic_values = (LIMIT - START) // STEP + 1
    final_total_combinations = num_alphas * num_betas * final_topic_values

    print(f"\nFINAL CONFIGURATION:")
    print(f"   - Topic range: {START} to {LIMIT} (step {STEP})")
    print(f"   - Alpha values: {num_alphas} options")
    print(f"   - Beta values: {num_betas} options")
    print(f"   - Total combinations: {final_total_combinations}")

    # Time estimation
    estimated_minutes = final_total_combinations * 0.05  # Approximately 3 seconds per model
    if estimated_minutes > 60:
        print(f"   - Estimated time: ~{estimated_minutes / 60:.1f} hours")
    else:
        print(f"   - Estimated time: ~{estimated_minutes:.1f} minutes")

    print("=" * 60)

    # Final confirmation if many combinations
    if final_total_combinations > 200:
        confirm_proceed = input(
            f"\nYou will evaluate {final_total_combinations} combinations. Continue? (y/n): ").strip().lower()
        if confirm_proceed not in ['y', 'yes']:
            print("Operation cancelled.")
            return

    # ---------- PREPROCESSING (spaCy) ----------
    print("Loading data...")
    df = joblib.load(RAW_PATH)
    df['cleaned'] = df['text'].astype(str).str.lower()

    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    STOPWORDS = nlp.Defaults.stop_words
    removal = ['ADV', 'PRON', 'CCONJ', 'PUNCT', 'PART', 'DET', 'ADP', 'SPACE', 'SYM']

    print("Lemmatizing and filtering texts...")
    texts = df['cleaned'].tolist()
    # Use a single process or adjust as needed; guard prevents spawn-related deadlocks
    docs = list(nlp.pipe(texts, batch_size=32, n_process=2))
    tokens = [
        [token.lemma_.lower() for token in doc if token.pos_ not in removal
         and len(token) > 1
         and not token.is_stop
         and token.lemma_.lower() not in STOPWORDS
         and token not in STOPWORDS
         and token.is_alpha]
        for doc in docs
    ]
    df['tokens'] = tokens

    # ---------- CORPUS AND DICTIONARY ----------
    print("Creating Gensim dictionary and corpus...")
    id2word = corpora.Dictionary(tokens)
    id2word.filter_extremes(no_below=5, no_above=0.7)
    corpus = [id2word.doc2bow(text) for text in tokens]

    # ---------- AUTOMATIC TOPIC SELECTION ----------
    def compute_coherence_values(dictionary, corpus, texts, limit, start, step, alpha, beta):
        coherence_values = []
        model_list = []
        model_parameters = []
        total_iterations = len(list(range(start, limit + 1, step))) * len(alpha) * len(beta)
        current_iteration = 0

        for num_topics in range(start, limit + 1, step):
            for a in alpha:
                for b in beta:
                    model = LdaModel(corpus=corpus,
                                     id2word=dictionary,
                                     num_topics=num_topics,
                                     random_state=0,
                                     chunksize=100,
                                     alpha=a,
                                     eta=b,
                                     passes=50,
                                     iterations=50,
                                     eval_every=None,
                                     minimum_probability=0.01,
                                     per_word_topics=True)
                    model_list.append(model)
                    coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
                    coherence_values.append(coherencemodel.get_coherence())
                    model_parameters.append({'num_topics': num_topics, 'alpha': a, 'eta': b})

                    # Display progress
                    current_iteration += 1
                    if current_iteration % 10 == 0 or current_iteration == total_iterations:
                        progress_percent = (current_iteration / total_iterations) * 100
                        print(f"\rProgress: {current_iteration}/{total_iterations} ({progress_percent:.1f}%)", end="",
                              flush=True)

        print()  # New line after progress
        return model_list, coherence_values, model_parameters

    print("Finding optimal number of topics and best hyperparameters...")
    print(f"Evaluating {final_total_combinations} combinations...")
    print("Progress: ", end="", flush=True)

    model_list, coherence_values, model_parameters = compute_coherence_values(
        dictionary=id2word,
        corpus=corpus,
        texts=tokens,
        start=START,
        limit=LIMIT,
        step=STEP,
        alpha=ALPHAS,
        beta=BETAS
    )

    # ---------- SAVE EVALUATION RESULTS ----------
    eval_results = pd.DataFrame(model_parameters)
    eval_results['coherence'] = coherence_values
    eval_results.to_csv(os.path.join(LDA_EVAL_DIR, "lda_coherence_grid.csv"), index=False)

    # Get the 4 best models based on coherence
    top_10_models = eval_results.nlargest(10, 'coherence').reset_index(drop=True)

    print("\n" + "=" * 80)
    print("TOP 10 BEST MODELS (by coherence)")
    print("=" * 80)

    # Display detailed information of each model with its topics
    for i, row in top_10_models.iterrows():
        model_idx = eval_results[eval_results['coherence'] == row['coherence']].index[0]
        current_model = model_list[model_idx]

        print(f"\n{i + 1}. MODEL #{i + 1}")
        print(f"   Topics: {row['num_topics']:2d} | Alpha: {str(row['alpha']):10s} | "
              f"Beta: {str(row['eta']):10s} | Coherence: {row['coherence']:.4f}")
        print("   " + "-" * 70)

        # Display most important terms for each topic
        for topic_num in range(row['num_topics']):
            topic_terms = [word for word, _ in current_model.show_topic(topic_num, topn=10)]
            print(f"   Topic {topic_num:2d}: {', '.join(topic_terms)}")

        print("   " + "-" * 70)

    print("\n" + "=" * 80)

    # Function to display detailed model information
    def show_detailed_model_info(model_idx, model_row, model_obj):
        print(f"\n{'=' * 80}")
        print(f"DETAILED MODEL INFORMATION - MODEL #{model_idx + 1}")
        print(f"{'=' * 80}")
        print(f"Configuration:")
        print(f"  - Number of topics: {model_row['num_topics']}")
        print(f"  - Alpha: {model_row['alpha']}")
        print(f"  - Beta: {model_row['eta']}")
        print(f"  - Coherence: {model_row['coherence']:.4f}")
        print(f"\nDetailed vocabulary by topic:")
        print("-" * 80)

        for topic_num in range(model_row['num_topics']):
            topic_words_with_prob = model_obj.show_topic(topic_num, topn=15)
            print(f"\nTOPIC {topic_num} (Top 15 words):")

            # Display words with probabilities
            words_str = ", ".join([f"{word}({prob:.3f})" for word, prob in topic_words_with_prob[:8]])
            print(f"  Main: {words_str}")

            # Display additional words
            additional_words = [word for word, _ in topic_words_with_prob[8:15]]
            if additional_words:
                print(f"  Additional: {', '.join(additional_words)}")

        print("=" * 80)

    # Allow user to choose the model
    while True:
        try:
            print("\nWhich model do you want to use for classification?")
            print("Options:")
            print("  1-10: Select model from the ranking")
            print("  d1-d10: View DETAILED information of the model (ex: 'd2' for model 2)")
            print("  Enter: Use the best model automatically")

            choice = input("Your choice: ").strip().lower()

            if choice == "":
                # Use best model automatically
                selected_idx = 0
                print("Using the best model automatically...")
                break
            elif choice in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']:
                selected_idx = int(choice) - 1
                print(f"You selected model #{choice}")
                break
            elif choice in ['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'd9', 'd10']:
                # Display detailed information
                detail_idx = int(choice[1]) - 1
                detail_row = top_10_models.iloc[detail_idx]
                detail_model_idx = eval_results[eval_results['coherence'] == detail_row['coherence']].index[0]
                detail_model = model_list[detail_model_idx]

                show_detailed_model_info(detail_idx, detail_row, detail_model)

                # Ask if user wants to select this model
                select_choice = input(f"\nDo you want to use this model #{choice[1]}? (y/n): ").strip().lower()
                if select_choice in ['y', 'yes']:
                    selected_idx = detail_idx
                    print(f"You selected model #{choice[1]}")
                    break
                else:
                    continue
            else:
                print("Please enter:")
                print("  - A number from 1 to 10 to select")
                print("  - 'd1', 'd2', ..., 'd10' to see details")
                print("  - Press Enter for automatic best model")
        except KeyboardInterrupt:
            print("\nOperation cancelled. Using the best model...")
            selected_idx = 0
            break
        except:
            print("Invalid input. Try again.")

    # Get the selected model
    selected_model_params = top_10_models.iloc[selected_idx]
    selected_model_idx = eval_results[eval_results['coherence'] == selected_model_params['coherence']].index[0]

    best_model = model_list[selected_model_idx]
    best_params = model_parameters[selected_model_idx]
    best_num_topics = best_params['num_topics']

    print(f"\n{'=' * 60}")
    print("SELECTED MODEL FOR CLASSIFICATION")
    print("=" * 60)
    print(f"Ranking: #{selected_idx + 1} of 10")
    print(f"Topics: {best_num_topics}")
    print(f"Alpha: {best_params['alpha']}")
    print(f"Beta: {best_params['eta']}")
    print(f"Coherence: {eval_results.iloc[selected_model_idx]['coherence']:.4f}")
    print("=" * 60)

    # Display topics of selected model
    print(f"\nTOPICS OF THE SELECTED MODEL:")
    for topic_num in range(best_num_topics):
        topic_terms = [word for word, _ in best_model.show_topic(topic_num, topn=10)]
        print(f"  Topic {topic_num:2d}: {', '.join(topic_terms)}")
    print("=" * 60)

    # Save model, dictionary, corpus and cleaned tokens
    best_model.save(os.path.join(LDA_EVAL_DIR, "lda_model"))
    id2word.save(os.path.join(LDA_EVAL_DIR, "dictionary.dict"))
    with open(os.path.join(LDA_EVAL_DIR, "corpus.pkl"), "wb") as f:
        pickle.dump(corpus, f)
    with open(os.path.join(LDA_EVAL_DIR, "tokens.pkl"), "wb") as f:
        pickle.dump(tokens, f)

    # Tag each document with main topic and distribution
    df['topic_dist'] = [best_model.get_document_topics(id2word.doc2bow(text)) for text in tokens]
    df['topic'] = [max(dist, key=lambda x: x[1])[0] for dist in df['topic_dist']]
    df.to_pickle(os.path.join(LDA_EVAL_DIR, "df_topic.pkl"))

    # Most relevant terms by topic
    top_terms_by_topic = {}
    for topic in range(best_num_topics):
        top_terms_by_topic[topic] = [w for w, _ in best_model.show_topic(topic, topn=30)]
    with open(os.path.join(LDA_EVAL_DIR, "top_terms_by_topic.pkl"), "wb") as f:
        pickle.dump(top_terms_by_topic, f)

    # Save information of the selected model
    selected_model_info = {
        'selected_model_rank': selected_idx + 1,
        'selected_model_params': best_params,
        'selected_model_coherence': eval_results.iloc[selected_model_idx]['coherence'],
        'top_10_models': top_10_models.to_dict('records'),
        'selection_method': 'manual' if choice != "" else 'automatic',
        'selected_topics_vocabulary': {}
    }

    # Save vocabulary of selected topics
    for topic_num in range(best_num_topics):
        topic_words = best_model.show_topic(topic_num, topn=20)
        selected_model_info['selected_topics_vocabulary'][f'topic_{topic_num}'] = {
            'words_with_probs': topic_words,
            'top_words': [word for word, _ in topic_words[:10]]
        }

    with open(os.path.join(LDA_EVAL_DIR, "selected_model_info.pkl"), "wb") as f:
        pickle.dump(selected_model_info, f)

    print(f"Topics and evaluation saved to: {LDA_EVAL_DIR}")

    # ---------- CLASSIFICATION PIPELINE ----------
    print("\n" + "=" * 50)
    print("STARTING CLASSIFICATION PIPELINE")
    print("=" * 50)

    # Create feature matrix (topic probability distribution)
    def create_topic_distribution_matrix(model, corpus, num_topics):
        """Create matrix where each row is the probability distribution of topics for a document"""
        topic_matrix = []

        for doc_topics in model.get_document_topics(corpus):
            # Initialize vector with zeros
            topic_vector = [0.0] * num_topics

            # Fill with probabilities of the topics present
            for topic_id, prob in doc_topics:
                topic_vector[topic_id] = prob

            topic_matrix.append(topic_vector)

        return np.array(topic_matrix)

    print("Creating topic distribution matrix...")
    X = create_topic_distribution_matrix(best_model, corpus, best_num_topics)

    # Get true labels (assuming 'type' or similar column exists)
    if 'type' in df.columns:
        y_original = df['type'].values
        target_column = 'type'
    elif 'category' in df.columns:
        y_original = df['category'].values
        target_column = 'category'
    elif 'target' in df.columns:
        y_original = df['target'].values
        target_column = 'target'
    else:
        print("Error: No label column found ('type', 'target' or 'category')")
        return

    # Encode categorical labels to numeric for XGBoost
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_original)

    # Save the mapping for future reference
    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

    print(f"Feature matrix: {X.shape}")
    print(f"Original classes: {np.unique(y_original)}")
    print(f"Encoded classes: {np.unique(y)}")
    print(f"Label mapping: {label_mapping}")
    print(f"Class distribution:")
    unique_original, counts = np.unique(y_original, return_counts=True)
    for label, count in zip(unique_original, counts):
        print(f"  {label}: {count}")

    # Create DataFrame with features and target
    feature_columns = [str(i) for i in range(best_num_topics)]
    df_classification = pd.DataFrame(X, columns=feature_columns)
    df_classification['target'] = y_original  # Use original labels in the CSV
    df_classification['target_encoded'] = y  # Also add the encoded ones

    # Save feature matrix
    df_classification.to_csv(os.path.join(LDA_EVAL_DIR, f"topic_distribution_matrix_{best_num_topics}_topics.csv"),
                             index=False)

    # ---------- CLASSIFICATION WITH CHOSEN CLASSIFIER ----------
    print(f"\nTraining classifier: {CLASSIFIER_TYPE.upper()}")

    if CLASSIFIER_TYPE == 'svm':
        # SVM configuration with Grid Search
        param_grid_svm = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'kernel': ['rbf', 'poly', 'sigmoid']
        }

        classifier = GridSearchCV(
            SVC(probability=True, random_state=42),
            param_grid_svm,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )

    elif CLASSIFIER_TYPE == 'xgboost':
        # XGBoost configuration with Grid Search - CORRECTED VERSION
        try:
            from xgboost import XGBClassifier

            # Check sklearn compatibility
            import sklearn
            sklearn_version = sklearn.__version__
            print(f"Sklearn version: {sklearn_version}")

            # Create base classifier
            base_xgb = XGBClassifier(
                random_state=42,
                eval_metric='logloss',
                verbosity=0  # Reduce verbosity to avoid warnings
            )

            # More conservative parameter grid
            param_grid_xgb = {
                'subsample': [0.4, 0.6, 0.8],
                'colsample_bytree': [0.4, 0.6, 0.8],
                'min_child_weight': [10, 30, 50],
                'max_depth': [10, 30, 100, 150, 200],
                'n_estimators': [50, 100, 150, 200],
                'reg_lambda': [0, 0.5, 1],
                'reg_alpha': [0, 0.5, 1]
            }
            # Create GridSearchCV with error handling
            try:
                classifier = GridSearchCV(
                    base_xgb,
                    param_grid_xgb,
                    cv=5,
                    scoring='accuracy',
                    n_jobs=-1,
                    verbose=1,
                    error_score='raise'
                )
            except Exception as e:
                print(f"Error creating GridSearchCV with XGBoost: {e}")
                print("Using XGBoost with default parameters...")
                classifier = XGBClassifier(
                    random_state=42,
                    eval_metric='logloss',
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1
                )

        except ImportError:
            print("XGBoost not installed. Installing...")
            import subprocess
            import sys
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])
                from xgboost import XGBClassifier

                classifier = XGBClassifier(
                    random_state=42,
                    eval_metric='logloss',
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1
                )
            except Exception as e:
                print(f"Error installing XGBoost: {e}")
                print("Switching to SVM as alternative...")
                CLASSIFIER_TYPE = 'svm'
                param_grid_svm = {
                    'C': [0.1, 1, 10],
                    'gamma': ['scale', 'auto'],
                    'kernel': ['rbf', 'poly']
                }
                classifier = GridSearchCV(
                    SVC(probability=True, random_state=42),
                    param_grid_svm,
                    cv=5,
                    scoring='accuracy',
                    n_jobs=-1,
                    verbose=1
                )

    # Train classifier with error handling
    print("Training with cross-validation...")
    try:
        classifier.fit(X, y)

        # If GridSearchCV, display best parameters
        if hasattr(classifier, 'best_params_'):
            print(f"\nBest parameters: {classifier.best_params_}")
            print(f"Best CV score: {classifier.best_score_:.4f}")

        # Predictions
        y_pred = classifier.predict(X)
        y_pred_proba = classifier.predict_proba(X)

    except Exception as e:
        print(f"Error during training: {e}")
        print("Trying with a simpler model...")

        # Fallback to a simpler model
        if CLASSIFIER_TYPE == 'xgboost':
            try:
                from xgboost import XGBClassifier
                classifier = XGBClassifier(
                    random_state=42,
                    eval_metric='logloss',
                    n_estimators=50,
                    max_depth=3,
                    learning_rate=0.1,
                    verbosity=0
                )
            except:
                from sklearn.ensemble import RandomForestClassifier
                classifier = RandomForestClassifier(
                    random_state=42,
                    n_estimators=50,
                    max_depth=10
                )
                CLASSIFIER_TYPE = 'random_forest'
        else:
            from sklearn.svm import SVC
            classifier = SVC(probability=True, random_state=42, C=1.0, gamma='scale')

        classifier.fit(X, y)
        y_pred = classifier.predict(X)
        y_pred_proba = classifier.predict_proba(X)

    # ---------- EVALUATION ----------
    print("\n" + "=" * 50)
    print("EVALUATION METRICS")
    print("=" * 50)

    # Accuracy
    accuracy = accuracy_score(y, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # Log Loss
    logloss = log_loss(y, y_pred_proba)
    print(f"Log Loss: {logloss:.4f}")

    # Mean Per Class Error
    cm = confusion_matrix(y, y_pred)
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    mean_per_class_error = 1 - np.mean(per_class_accuracy)
    print(f"Mean Per Class Error: {mean_per_class_error:.4f}")

    # AUC (only for binary classification)
    unique_classes_original = np.unique(y_original)
    unique_classes_encoded = np.unique(y)
    if len(unique_classes_encoded) == 2:
        auc = roc_auc_score(y, y_pred_proba[:, 1])
        print(f"AUC: {auc:.4f}")
    else:
        try:
            auc = roc_auc_score(y, y_pred_proba, multi_class='ovr', average='macro')
            print(f"AUC (macro): {auc:.4f}")
        except:
            print("AUC: Not calculable for this dataset")

    # Confusion Matrix - display with original labels
    print(f"\nConfusion Matrix:")
    print("Original classes:", unique_classes_original)
    print("Encoded classes:", unique_classes_encoded)
    print(cm)

    # Decode predictions for report
    y_pred_original = label_encoder.inverse_transform(y_pred)

    # Detailed classification report with original labels
    print(f"\nClassification Report:")
    print(classification_report(y_original, y_pred_original))


    # ---------- SAVE RESULTS ----------
    results_dict = {
        'classifier_type': CLASSIFIER_TYPE,
        'best_params': getattr(classifier, 'best_params_', 'N/A'),
        'best_cv_score': getattr(classifier, 'best_score_', 'N/A'),
        'accuracy': accuracy,
        'log_loss': logloss,
        'mean_per_class_error': mean_per_class_error,
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(y_original, y_pred_original, output_dict=True),
        'num_topics': best_num_topics,
        'features_shape': X.shape,
        'classes_original': unique_original.tolist(),
        'classes_encoded': unique_classes_encoded.tolist(),
        'label_mapping': label_mapping
    }

    with open(os.path.join(LDA_EVAL_DIR, f"classification_results_{CLASSIFIER_TYPE}.pkl"), "wb") as f:
        pickle.dump(results_dict, f)

    joblib.dump(classifier, os.path.join(LDA_EVAL_DIR, f"classifier_{CLASSIFIER_TYPE}_model.pkl"))
    joblib.dump(label_encoder, os.path.join(LDA_EVAL_DIR, f"label_encoder.pkl"))

    predictions_df = pd.DataFrame({
        'true_label': y_original,
        'predicted_label': y_pred_original,
        'true_label_encoded': y,
        'predicted_label_encoded': y_pred,
        'max_probability': np.max(y_pred_proba, axis=1)
    })

    for i, class_name in enumerate(label_encoder.classes_):
        predictions_df[f'prob_{class_name}'] = y_pred_proba[:, i]

    predictions_df.to_csv(os.path.join(LDA_EVAL_DIR, f"predictions_{CLASSIFIER_TYPE}.csv"), index=False)

    print(f"\nClassification results saved to: {LDA_EVAL_DIR}")
    print(f"- Model: classifier_{CLASSIFIER_TYPE}_model.pkl")
    print(f"- Label Encoder: label_encoder.pkl")
    print(f"- Results: classification_results_{CLASSIFIER_TYPE}.pkl")
    print(f"- Predictions: predictions_{CLASSIFIER_TYPE}.csv")
    print(f"- Feature matrix: topic_distribution_matrix_{best_num_topics}_topics.csv")

    print("\n" + "=" * 50)
    print("PROCESS COMPLETED SUCCESSFULLY")
    print("=" * 50)


if __name__ == "__main__":
    mp.set_start_method("fork")
    main()