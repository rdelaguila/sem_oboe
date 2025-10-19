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

    # ---------------- ARGUMENTOS DE LÍNEA DE COMANDOS ----------------
    parser = argparse.ArgumentParser(description='LDA Topic Modeling with Classification')
    parser.add_argument('--classifier', type=str, choices=['svm', 'xgboost'],
                        default='xgboost', help='Choose classifier: svm or xgboost')
    parser.add_argument('--raw_path', type=str, default="data/processed/amazon/amazon_processed_semantic.pkl",
                        help='Path to raw data file')
    parser.add_argument('--eval_dir', type=str, default="data/lda_eval/amazon",
                        help='Directory to save LDA evaluation results')
    parser.add_argument('--max_topics', type=int, default=None,
                        help='Maximum number of topics to test (overrides default)')

    # Parsear argumentos con manejo de errores
    try:
        args = parser.parse_args()
    except SystemExit:
        # Si no hay argumentos o hay error en argumentos, usar valores por defecto
        print("No se proporcionaron argumentos válidos. Usando valores por defecto:")
        args = argparse.Namespace(
            classifier='xgboost',
            raw_path="../olds/data/corpus_raw/amazon/corpus_raw",
            eval_dir="../olds/data/lda_eval/amazon",
            max_topics=None
        )

    # ---------------- CONFIG ----------------
    RAW_PATH = args.raw_path
    LDA_EVAL_DIR = args.eval_dir
    CLASSIFIER_TYPE = args.classifier
    MAX_TOPICS_ARG = args.max_topics

    # Mostrar configuración actual
    print("=" * 50)
    print("CONFIGURACIÓN DEL PIPELINE")
    print("=" * 50)
    print(f"Clasificador: {CLASSIFIER_TYPE}")
    print(f"Archivo de datos: {RAW_PATH}")
    print(f"Directorio de salida: {LDA_EVAL_DIR}")
    if MAX_TOPICS_ARG:
        print(f"Máximo de tópicos (argumento): {MAX_TOPICS_ARG}")
    print("=" * 50)

    # Verificar que el archivo de datos existe
    if not os.path.exists(RAW_PATH):
        print(f"ERROR: No se encontró el archivo de datos en: {RAW_PATH}")
        print("Por favor, verifica la ruta o ejecuta primero el script de preprocesamiento.")
        return

    os.makedirs(LDA_EVAL_DIR, exist_ok=True)

    # ---------------- CONFIGURACIÓN DE PARÁMETROS DE TÓPICOS ----------------
    print("\n" + "=" * 60)
    print("CONFIGURACIÓN DE PARÁMETROS PARA TOPIC MODELING")
    print("=" * 60)

    # Alpha parameter


    ALPHAS = list(np.arange(0.009, 0.15, 0.06))
    ALPHAS.append('symmetric')
    ALPHAS.append('asymmetric')
    # Beta parameter
    BETAS = list(np.arange(0.5, 1, 0.04))
    BETAS.append('symmetric')

    # Configuración del número máximo de topics
    DEFAULT_MAX_TOPICS = 15
    START = 2
    STEP = 1

    # Calcular número de combinaciones de hiperparámetros
    num_alphas = len(ALPHAS)
    num_betas = len(BETAS)

    # Si se proporcionó un argumento, usarlo
    if MAX_TOPICS_ARG:
        LIMIT = MAX_TOPICS_ARG
        print(f"Usando máximo de tópicos desde argumentos: {LIMIT}")
    else:
        # Solicitar al usuario el número máximo de topics
        print(f"Configuración actual: evaluar desde {START} hasta {DEFAULT_MAX_TOPICS} tópicos (paso {STEP})")
        print("Esto evaluará combinaciones de hiperparámetros para cada número de tópicos.")

        # Calcular número total de combinaciones
        num_topic_values = (DEFAULT_MAX_TOPICS - START) // STEP + 1
        total_combinations = num_alphas * num_betas * num_topic_values

        print(f"Total de combinaciones a evaluar: {total_combinations}")
        print(f"(Alpha: {num_alphas} × Beta: {num_betas} × Topics: {num_topic_values})")

        while True:
            try:
                user_input = input(
                    f"\n¿Número máximo de tópicos a evaluar? (Enter = {DEFAULT_MAX_TOPICS}, mín=2): ").strip()

                if user_input == "":
                    LIMIT = DEFAULT_MAX_TOPICS
                    print(f"Usando valor por defecto: {LIMIT} tópicos máximo")
                    break
                elif user_input.isdigit():
                    user_limit = int(user_input)
                    if user_limit < 2:
                        print("El número mínimo de tópicos debe ser 2. Intenta de nuevo.")
                        continue
                    elif user_limit > 50:
                        confirm = input(
                            f"¿Confirmas evaluar hasta {user_limit} tópicos? Esto puede tomar mucho tiempo (s/n): ").strip().lower()
                        if confirm in ['s', 'si', 'y', 'yes']:
                            LIMIT = user_limit
                            break
                        else:
                            continue
                    else:
                        LIMIT = user_limit
                        break
                else:
                    print("Por favor, ingresa un número válido o presiona Enter.")

            except KeyboardInterrupt:
                print(f"\nUsando valor por defecto: {DEFAULT_MAX_TOPICS}")
                LIMIT = DEFAULT_MAX_TOPICS
                break
            except:
                print("Entrada inválida. Intenta de nuevo.")

    # Recalcular y mostrar configuración final
    final_topic_values = (LIMIT - START) // STEP + 1
    final_total_combinations = num_alphas * num_betas * final_topic_values

    print(f"\nCONFIGURACIÓN FINAL:")
    print(f"   - Rango de tópicos: {START} a {LIMIT} (paso {STEP})")
    print(f"   - Valores de Alpha: {num_alphas} opciones")
    print(f"   - Valores de Beta: {num_betas} opciones")
    print(f"   - Total combinaciones: {final_total_combinations}")

    # Estimación de tiempo
    estimated_minutes = final_total_combinations * 0.05  # Aproximadamente 3 segundos por modelo
    if estimated_minutes > 60:
        print(f"   - Tiempo estimado: ~{estimated_minutes / 60:.1f} horas")
    else:
        print(f"   - Tiempo estimado: ~{estimated_minutes:.1f} minutos")

    print("=" * 60)

    # Confirmación final si hay muchas combinaciones
    if final_total_combinations > 200:
        confirm_proceed = input(
            f"\nVas a evaluar {final_total_combinations} combinaciones. ¿Continuar? (s/n): ").strip().lower()
        if confirm_proceed not in ['s', 'si', 'y', 'yes']:
            print("Operación cancelada.")
            return

    # ---------------- PREPROCESAMIENTO (spaCy) ----------------
    print("Cargando datos...")
    df = joblib.load(RAW_PATH)
    df['cleaned'] = df['text'].astype(str).str.lower()

    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    STOPWORDS = nlp.Defaults.stop_words
    removal = ['ADV', 'PRON', 'CCONJ', 'PUNCT', 'PART', 'DET', 'ADP', 'SPACE', 'SYM']

    print("Lemmatizando y filtrando textos...")
    texts = df['cleaned'].tolist()
    # Use a single process or adjust as needed; guard prevents spawn-related deadlocks
    docs = list(nlp.pipe(texts, batch_size=32, n_process=2))
    tokens = [
        [token.lemma_.lower() for token in doc if token.pos_ not in removal

         and len(token)>1
         and not token.is_stop
         and token.lemma_.lower() not in STOPWORDS
         and token not in STOPWORDS
         and token.is_alpha]
        for doc in docs
    ]
    df['tokens'] = tokens

    # ---------------- CORPUS Y DICCIONARIO ----------------
    print("Creando diccionario y corpus Gensim...")
    id2word = corpora.Dictionary(tokens)
    id2word.filter_extremes(no_below=5, no_above=0.7)
    corpus = [id2word.doc2bow(text) for text in tokens]

    # ---------------- SELECCIÓN AUTOMÁTICA DE TOPICSTOPICS ----------------
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
                                    # update_every=0,  # No actualizar hasta completar época
                                     eval_every=None,
                                     minimum_probability = 0.01,
                                     per_word_topics=True)
                    model_list.append(model)
                    coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
                    coherence_values.append(coherencemodel.get_coherence())
                    model_parameters.append({'num_topics': num_topics, 'alpha': a, 'eta': b})

                    # Mostrar progreso
                    current_iteration += 1
                    if current_iteration % 10 == 0 or current_iteration == total_iterations:
                        progress_percent = (current_iteration / total_iterations) * 100
                        print(f"\rProgreso: {current_iteration}/{total_iterations} ({progress_percent:.1f}%)", end="",
                              flush=True)

        print()  # Nueva línea después del progreso
        return model_list, coherence_values, model_parameters

    print("Buscando número óptimo de tópicos y mejores hiperparámetros...")
    print(f"Evaluando {final_total_combinations} combinaciones...")
    print("Progreso: ", end="", flush=True)

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

    # ---------------- GUARDAR RESULTADOS DE EVALUACIÓN ----------------
    eval_results = pd.DataFrame(model_parameters)
    eval_results['coherence'] = coherence_values
    eval_results.to_csv(os.path.join(LDA_EVAL_DIR, "lda_coherence_grid.csv"), index=False)

    # Obtener los 4 mejores modelos basados en coherencia
    top_10_models = eval_results.nlargest(10, 'coherence').reset_index(drop=True)

    print("\n" + "=" * 80)
    print("TOP 10 MEJORES MODELOS (por coherencia)")
    print("=" * 80)

    # Mostrar información detallada de cada modelo con sus topics
    for i, row in top_10_models.iterrows():
        model_idx = eval_results[eval_results['coherence'] == row['coherence']].index[0]
        current_model = model_list[model_idx]

        print(f"\n{i + 1}. MODELO #{i + 1}")
        print(f"   Topics: {row['num_topics']:2d} | Alpha: {str(row['alpha']):10s} | "
              f"Beta: {str(row['eta']):10s} | Coherence: {row['coherence']:.4f}")
        print("   " + "-" * 70)

        # Mostrar los términos más importantes para cada tópico
        for topic_num in range(row['num_topics']):
            topic_terms = [word for word, _ in current_model.show_topic(topic_num, topn=10)]
            print(f"   Tópico {topic_num:2d}: {', '.join(topic_terms)}")

        print("   " + "-" * 70)

    print("\n" + "=" * 80)

    # Función para mostrar información detallada de un modelo específico
    def show_detailed_model_info(model_idx, model_row, model_obj):
        print(f"\n{'=' * 80}")
        print(f"INFORMACIÓN DETALLADA - MODELO #{model_idx + 1}")
        print(f"{'=' * 80}")
        print(f"Configuración:")
        print(f"  - Número de tópicos: {model_row['num_topics']}")
        print(f"  - Alpha: {model_row['alpha']}")
        print(f"  - Beta: {model_row['eta']}")
        print(f"  - Coherencia: {model_row['coherence']:.4f}")
        print(f"\nVocabulario detallado por tópico:")
        print("-" * 80)

        for topic_num in range(model_row['num_topics']):
            topic_words_with_prob = model_obj.show_topic(topic_num, topn=15)
            print(f"\nTÓPICO {topic_num} (Top 15 palabras):")

            # Mostrar palabras con sus probabilidades
            words_str = ", ".join([f"{word}({prob:.3f})" for word, prob in topic_words_with_prob[:8]])
            print(f"  Principales: {words_str}")

            # Mostrar palabras adicionales
            additional_words = [word for word, _ in topic_words_with_prob[8:15]]
            if additional_words:
                print(f"  Adicionales: {', '.join(additional_words)}")

        print("=" * 80)

    # Permitir al usuario elegir el modelo
    while True:
        try:
            print("\n¿Qué modelo quieres usar para la clasificación?")
            print("Opciones:")
            print("  1-10: Seleccionar modelo del ranking")
            print("  d1-d10: Ver información DETALLADA del modelo (ej: 'd2' para modelo 2)")
            print("  Enter: Usar el mejor modelo automáticamente")

            choice = input("Tu elección: ").strip().lower()

            if choice == "":
                # Usar el mejor modelo automáticamente
                selected_idx = 0
                print("Usando el mejor modelo automáticamente...")
                break
            elif choice in ['1', '2', '3', '4','5','6','7','8','9','10']:
                selected_idx = int(choice) - 1
                print(f"Has seleccionado el modelo #{choice}")
                break
            elif choice in ['d1', 'd2', 'd3', 'd4','d5', 'd6', 'd7', 'd8','d9', 'd10']:
                # Mostrar información detallada
                detail_idx = int(choice[1]) - 1
                detail_row = top_10_models.iloc[detail_idx]
                detail_model_idx = eval_results[eval_results['coherence'] == detail_row['coherence']].index[0]
                detail_model = model_list[detail_model_idx]

                show_detailed_model_info(detail_idx, detail_row, detail_model)

                # Preguntar si quiere seleccionar este modelo
                select_choice = input(f"\n¿Quieres usar este modelo #{choice[1]}? (s/n): ").strip().lower()
                if select_choice in ['s', 'si', 'y', 'yes']:
                    selected_idx = detail_idx
                    print(f"Has seleccionado el modelo #{choice[1]}")
                    break
                else:
                    continue
            else:
                print("Por favor, ingresa:")
                print("  - Un número del 1 al 10 para seleccionar")
                print("  - 'd1', 'd2', ...,  'd10' para ver detalles")
                print("  - Presiona Enter para usar el mejor automáticamente")
        except KeyboardInterrupt:
            print("\nOperación cancelada. Usando el mejor modelo...")
            selected_idx = 0
            break
        except:
            print("Entrada inválida. Por favor, intenta de nuevo.")

    # Obtener el modelo seleccionado
    selected_model_params = top_10_models.iloc[selected_idx]
    selected_model_idx = eval_results[eval_results['coherence'] == selected_model_params['coherence']].index[0]

    best_model = model_list[selected_model_idx]
    best_params = model_parameters[selected_model_idx]
    best_num_topics = best_params['num_topics']

    print(f"\n{'=' * 60}")
    print("MODELO SELECCIONADO PARA CLASIFICACIÓN")
    print("=" * 60)
    print(f"Ranking: #{selected_idx + 1} de 10")
    print(f"Topics: {best_num_topics}")
    print(f"Alpha: {best_params['alpha']}")
    print(f"Beta: {best_params['eta']}")
    print(f"Coherence: {eval_results.iloc[selected_model_idx]['coherence']:.4f}")
    print("=" * 60)

    # Mostrar tópicos del modelo seleccionado
    print(f"\nTÓPICOS DEL MODELO SELECCIONADO:")
    for topic_num in range(best_num_topics):
        topic_terms = [word for word, _ in best_model.show_topic(topic_num, topn=10)]
        print(f"  Tópico {topic_num:2d}: {', '.join(topic_terms)}")
    print("=" * 60)

    # Guarda el modelo, diccionario, corpus y tokens limpios
    best_model.save(os.path.join(LDA_EVAL_DIR, "lda_model"))
    id2word.save(os.path.join(LDA_EVAL_DIR, "dictionary.dict"))
    with open(os.path.join(LDA_EVAL_DIR, "corpus.pkl"), "wb") as f:
        pickle.dump(corpus, f)
    with open(os.path.join(LDA_EVAL_DIR, "tokens.pkl"), "wb") as f:
        pickle.dump(tokens, f)

    # Etiqueta cada doc con topic principal y distribución
    df['topic_dist'] = [best_model.get_document_topics(id2word.doc2bow(text)) for text in tokens]
    df['topic'] = [max(dist, key=lambda x: x[1])[0] for dist in df['topic_dist']]
    df.to_pickle(os.path.join(LDA_EVAL_DIR, "df_topic.pkl"))

    # Términos más relevantes por tópico
    top_terms_by_topic = {}
    for topic in range(best_num_topics):
        top_terms_by_topic[topic] = [w for w, _ in best_model.show_topic(topic, topn=30)]
    with open(os.path.join(LDA_EVAL_DIR, "top_terms_by_topic.pkl"), "wb") as f:
        pickle.dump(top_terms_by_topic, f)

    # Guardar información del modelo seleccionado
    selected_model_info = {
        'selected_model_rank': selected_idx + 1,
        'selected_model_params': best_params,
        'selected_model_coherence': eval_results.iloc[selected_model_idx]['coherence'],
        'top_10_models': top_10_models.to_dict('records'),
        'selection_method': 'manual' if choice != "" else 'automatic',
        'selected_topics_vocabulary': {}
    }

    # Guardar vocabulario de los tópicos seleccionados
    for topic_num in range(best_num_topics):
        topic_words = best_model.show_topic(topic_num, topn=20)
        selected_model_info['selected_topics_vocabulary'][f'topic_{topic_num}'] = {
            'words_with_probs': topic_words,
            'top_words': [word for word, _ in topic_words[:10]]
        }

    with open(os.path.join(LDA_EVAL_DIR, "selected_model_info.pkl"), "wb") as f:
        pickle.dump(selected_model_info, f)

    print(f"Tópicos y evaluación guardados en: {LDA_EVAL_DIR}")

    # ---------------- PIPELINE DE CLASIFICACIÓN ----------------
    print("\n" + "=" * 50)
    print("INICIANDO PIPELINE DE CLASIFICACIÓN")
    print("=" * 50)

    # Crear matriz de características (distribución de probabilidad de topics)
    def create_topic_distribution_matrix(model, corpus, num_topics):
        """Crea matriz donde cada fila es la distribución de probabilidad de topics para un documento"""
        topic_matrix = []

        for doc_topics in model.get_document_topics(corpus):
            # Inicializar vector con ceros
            topic_vector = [0.0] * num_topics

            # Llenar con las probabilidades de los topics presentes
            for topic_id, prob in doc_topics:
                topic_vector[topic_id] = prob

            topic_matrix.append(topic_vector)

        return np.array(topic_matrix)

    print("Creando matriz de distribución de topics...")
    X = create_topic_distribution_matrix(best_model, corpus, best_num_topics)

    # Obtener etiquetas verdaderas (asumiendo que existe columna 'type' o similar)
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
        print("Error: No se encontró columna de etiquetas ('type', 'target' o 'category')")
        return

    # Codificar etiquetas categóricas a numéricas para XGBoost
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_original)

    # Guardar el mapeo para futura referencia
    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

    print(f"Matriz de características: {X.shape}")
    print(f"Clases originales: {np.unique(y_original)}")
    print(f"Clases codificadas: {np.unique(y)}")
    print(f"Mapeo de etiquetas: {label_mapping}")
    print(f"Distribución de clases:")
    unique_original, counts = np.unique(y_original, return_counts=True)
    for label, count in zip(unique_original, counts):
        print(f"  {label}: {count}")

    # Crear DataFrame con características y target
    feature_columns = [str(i) for i in range(best_num_topics)]
    df_classification = pd.DataFrame(X, columns=feature_columns)
    df_classification['target'] = y_original  # Usar etiquetas originales en el CSV
    df_classification['target_encoded'] = y  # Añadir también las codificadas

    # Guardar matriz de características
    df_classification.to_csv(os.path.join(LDA_EVAL_DIR, f"topic_distribution_matrix_{best_num_topics}_topics.csv"),
                             index=False)

    # ---------------- CLASIFICACIÓN CON EL MODELO ELEGIDO ----------------
    print(f"\nEntrenando clasificador: {CLASSIFIER_TYPE.upper()}")

    if CLASSIFIER_TYPE == 'svm':
        # Configuración SVM con Grid Search
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
        # Configuración XGBoost con Grid Search - VERSIÓN CORREGIDA
        try:
            from xgboost import XGBClassifier

            # Verificar compatibilidad con sklearn
            import sklearn
            sklearn_version = sklearn.__version__
            print(f"Sklearn version: {sklearn_version}")

            # Crear el clasificador base
            base_xgb = XGBClassifier(
                random_state=42,
                eval_metric='logloss',
                verbosity=0  # Reducir verbosity para evitar warnings
            )

            # Grid de parámetros más conservador

            param_grid_xgb = {
                'subsample': [ 0.4, 0.6, 0.8],  # sample_rate en H2O
                'colsample_bytree': [0.4,  0.6, 0.8],  # col_sample_rate en H2O
                'min_child_weight': [10, 30, 50],  # min_rows en H2O
                'max_depth': [10, 30, 100, 150, 200],  # max_depth (igual)
                'n_estimators': [50, 100, 150, 200],  # ntrees en H2O
                'reg_lambda': [0,  0.5, 1],  # reg_lambda (igual)
                'reg_alpha': [0,  0.5,  1]  # reg_alpha (igual)
            }
            # Crear GridSearchCV con manejo de errores
            try:
                classifier = GridSearchCV(
                    base_xgb,
                    param_grid_xgb,
                    cv=5,
                    scoring='accuracy',
                    n_jobs=-1,  # Usar 1 job para evitar problemas de paralelización
                    verbose=1,
                    error_score='raise'  # Para debuggear mejor los errores
                )
            except Exception as e:
                print(f"Error creando GridSearchCV con XGBoost: {e}")
                print("Usando XGBoost con parámetros por defecto...")
                classifier = XGBClassifier(
                    random_state=42,
                    eval_metric='logloss',
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1
                )

        except ImportError:
            print("XGBoost no está instalado. Instalando...")
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
                print(f"Error instalando XGBoost: {e}")
                print("Cambiando a SVM como alternativa...")
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

    # Entrenar el clasificador con manejo de errores
    print("Entrenando con validación cruzada...")
    try:
        classifier.fit(X, y)

        # Si es GridSearchCV, mostrar mejores parámetros
        if hasattr(classifier, 'best_params_'):
            print(f"\nMejores parámetros: {classifier.best_params_}")
            print(f"Mejor score CV: {classifier.best_score_:.4f}")

        # Predicciones
        y_pred = classifier.predict(X)
        y_pred_proba = classifier.predict_proba(X)

    except Exception as e:
        print(f"Error durante el entrenamiento: {e}")
        print("Intentando con un modelo más simple...")

        # Fallback a un modelo simple
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

    # ---------------- EVALUACIÓN ----------------
    print("\n" + "=" * 50)
    print("MÉTRICAS DE EVALUACIÓN")
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

    # AUC (solo para clasificación binaria)
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
            print("AUC: No calculable para este dataset")

    # Matriz de Confusión - mostrar con etiquetas originales
    print(f"\nMatriz de Confusión:")
    print("Clases originales:", unique_classes_original)
    print("Clases codificadas:", unique_classes_encoded)
    print(cm)

    # Decodificar predicciones para el reporte
    y_pred_original = label_encoder.inverse_transform(y_pred)

    # Reporte de clasificación detallado con etiquetas originales
    print(f"\nReporte de Clasificación:")
    print(classification_report(y_original, y_pred_original))

    # ---------------- GUARDAR RESULTADOS DE CLASIFICACIÓN ----------------
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
        'classes_original': unique_classes_original.tolist(),
        'classes_encoded': unique_classes_encoded.tolist(),
        'label_mapping': label_mapping
    }

    # Añadir AUC si está disponible
    if len(unique_classes_encoded) == 2:
        results_dict['auc'] = auc
    elif len(unique_classes_encoded) > 2:
        try:
            results_dict['auc_macro'] = auc
        except:
            results_dict['auc_macro'] = None

    # Guardar resultados
    with open(os.path.join(LDA_EVAL_DIR, f"classification_results_{CLASSIFIER_TYPE}.pkl"), "wb") as f:
        pickle.dump(results_dict, f)

    # Guardar modelo entrenado y el label encoder
    joblib.dump(classifier, os.path.join(LDA_EVAL_DIR, f"classifier_{CLASSIFIER_TYPE}_model.pkl"))
    joblib.dump(label_encoder, os.path.join(LDA_EVAL_DIR, f"label_encoder.pkl"))

    # Guardar predicciones con etiquetas originales
    predictions_df = pd.DataFrame({
        'true_label': y_original,
        'predicted_label': y_pred_original,
        'true_label_encoded': y,
        'predicted_label_encoded': y_pred,
        'max_probability': np.max(y_pred_proba, axis=1)
    })

    # Añadir probabilidades para cada clase (usando etiquetas originales)
    for i, class_name in enumerate(label_encoder.classes_):
        predictions_df[f'prob_{class_name}'] = y_pred_proba[:, i]

    predictions_df.to_csv(os.path.join(LDA_EVAL_DIR, f"predictions_{CLASSIFIER_TYPE}.csv"), index=False)

    print(f"\nResultados de clasificación guardados en: {LDA_EVAL_DIR}")
    print(f"- Modelo: classifier_{CLASSIFIER_TYPE}_model.pkl")
    print(f"- Label Encoder: label_encoder.pkl")
    print(f"- Resultados: classification_results_{CLASSIFIER_TYPE}.pkl")
    print(f"- Predicciones: predictions_{CLASSIFIER_TYPE}.csv")
    print(f"- Matriz de características: topic_distribution_matrix_{best_num_topics}_topics.csv")

    print("\n" + "=" * 50)
    print("PROCESO COMPLETADO EXITOSAMENTE")
    print("=" * 50)


if __name__ == "__main__":
    # En macOS, usar 'fork' evita deadlocks con spawn
    mp.set_start_method("fork")
    main()