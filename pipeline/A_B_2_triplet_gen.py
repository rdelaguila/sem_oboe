#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import pandas as pd
import stanza
from stanza.server import CoreNLPClient
import joblib
import dask.dataframe as dd
import traceback
import importlib
from time import sleep
from utils.triplet_manager_lib import TripletManager,TripletGenerator,ValidadorTripletas


# ============================================================================
# FUNCIONES AUXILIARES GLOBALES
# ============================================================================

def devolver_tripletas(sentence, frase, pos, tripletgenerator):
    """Extrae tripletas de una oración usando diferentes métodos"""
    tripletas = [tripletgenerator.encapsular((triple.subject, triple.relation, triple.object), True)
                 for triple in sentence.openieTriple]


    if len(tripletas) == 0:
        tripleta = tripletgenerator.triplet_extraction(frase)
        if tripleta is None:
            return tripletas

    if tripletgenerator._detect_adj_nou(pos)[0] != -1:
        tripleta = tripletgenerator.generar_tripleta_adj_noun(pos)
    if tripleta is not None:
        tripleta = tripletgenerator.encapsular(tripleta, True)
        if type(tripleta) == list:
            tripletas.extend(tripleta)
        else:
            tripletas.append(tripleta)

    if tripletgenerator._detect_nn_nnp(pos)[0] != -1:
        tripleta = tripletgenerator.generar_tripleta_nn_nnp(pos)
        if tripleta is not None:
            tripleta = tripletgenerator.encapsular(tripleta, True)
            if type(tripleta) == list:
                tripletas.extend(tripleta)
            else:
                tripletas.append(tripleta)

    if tripletgenerator._detect_nn_of_nn(pos)[0] != -1:
        tripleta = tripletgenerator.generar_tripleta_nn_of_nn(pos)
        if tripleta is not None:
            tripleta = tripletgenerator.encapsular(tripleta, True)
            if type(tripleta) == list:
                tripletas.extend(tripleta)
            else:
                tripletas.append(tripleta)

    if tripletgenerator._detect_nn_place_nn(pos)[0] != -1:
        tripleta = tripletgenerator.generar_tripleta_nn_place_nn(pos)
        if tripleta is not None:
            tripleta = tripletgenerator.encapsular(tripleta, True)
            if type(tripleta) == list:
                tripletas.extend(tripleta)
            else:
                tripletas.append(tripleta)

    tripleta = tripletgenerator.generar_tripleta_adjectives_nn_adjs(pos)
    if tripleta is not None:
        tripleta = tripletgenerator.encapsular(tripleta, True)
        if type(tripleta) == list:
            tripletas.extend(tripleta)
        else:
            tripletas.append(tripleta)

    return tripletas


def devolver_pos(sentence):
    """Extrae información POS de una oración"""
    diccionario = dict()
    for posicion in (sentence.ListFields()[0][1]):
        diccionario[posicion.word] = posicion.pos
    return diccionario


# Función para usar en el procesamiento
def anotar_row_optimizado(row, client, sleep_time, enable_pos_filtering=True, debug_rechazadas = True):
    """
    Versión optimizada con validación básica + POS
    """
    tripletgenerator = TripletGenerator()
    validador = ValidadorTripletas()  # Crear instancia del validador
    resultado = []

    # Obtener información básica
    topic = row.get('topic')

    # Procesar el texto
    text_column = ''
    for text_col in ['text_coref', 'coref_text', 'text', 'news']:
        if text_col in row and row[text_col]:
            text_column = str(row[text_col])
            break

    if not text_column:
        return resultado

    for frase in text_column.split('.'):
        if not frase.strip():
            continue
        try:
            anotacion = client.annotate(frase)
            sleep(sleep_time)

            for sentence in anotacion.sentence:
                d = devolver_pos(sentence)
                t = devolver_tripletas(sentence, frase, d, tripletgenerator)

                if enable_pos_filtering:
                    # Aplicar validación con debug
                    tripletas_validas = []
                    for tripleta in t:
                        if debug_rechazadas:
                            es_valida, explicacion = validador.validacion_rapida_con_pos(tripleta, d, debug=True)
                            if es_valida:
                                tripletas_validas.append(tripleta)
                                print(
                                    f"✓ ACEPTADA: {tripleta.get('subject')} | {tripleta.get('relation')} | {tripleta.get('object')}")
                                print(f"    Razón: {explicacion}")
                            else:
                                print(
                                    f"✗ RECHAZADA: {tripleta.get('subject')} | {tripleta.get('relation')} | {tripleta.get('object')}")
                                print(f"    Razón: {explicacion}")
                        else:
                            if validador.validacion_rapida_con_pos(tripleta, d):
                                tripletas_validas.append(tripleta)

                    resultado.extend(tripletas_validas)
                else:
                    resultado.extend(t)

        except Exception as e:
                print(f"Error procesando frase '{frase[:50]}...': {e}")
                continue

        return resultado


def transformar_partition_optimizado(df, client, sleep_time, enable_pos_filtering, debug_rechazadas):
    """
    Transforma una partición del dataframe aplicando anotar_row_optimizado a cada fila
    """
    return df.apply(
        lambda row: anotar_row_optimizado(row, client, sleep_time, enable_pos_filtering, debug_rechazadas),
        axis=1
    )

def anotar_row(row, client, sleep_time, enable_filtering, filter_method):
    """
    Procesa una fila del dataframe y extrae tripletas
    """
    tripletgenerator = TripletGenerator()
    resultado = []

    # Debug: mostrar columnas disponibles en la primera fila
    #if not hasattr(anotar_row, '_debug_shown'):
    #    print(f"DEBUG: Columnas disponibles: {list(row.index)}")
    #    for col in ['topic', 'new_target', 'target', 'entidades', 'entidades_dbpedia',
    #                'entidades_dbpedia_simplificadas']:
    #        if col in row:
    #            print(f"DEBUG: {col} = {type(row[col])} - {str(row[col])[:100]}...")
    #    anotar_row._debug_shown = True

    # Obtener información de tópicos - buscar diferentes nombres de columna
    topic = row.get('topic')

    # Obtener datos para filtrado si está habilitado
    tripletmanager = None
    ner = set()
    dbpedia = []

    if enable_filtering:
        tripletmanager = TripletManager()

        # Obtener entidades NER - buscar diferentes nombres de columna
        if filter_method in ['ner_only', 'both']:
            for ner_col in ['entidades', 'ner', 'tokens']:
                if ner_col in row and row[ner_col]:
                    ner = row[ner_col]
                    if isinstance(ner, str):
                        ner = set([ner])
                    elif not isinstance(ner, set):
                        ner = set(ner) if ner else set()
                    break

        # Obtener entidades DBpedia - buscar diferentes nombres de columna
        if filter_method in ['dbpedia_only', 'both']:
            for dbpedia_col in ['entidades_dbpedia_simplificadas', 'entidades_dbpedia']:
                if dbpedia_col in row and row[dbpedia_col]:
                    entidades_dbpedia = row[dbpedia_col]

                    # Manejar diferentes tipos de datos
                    if isinstance(entidades_dbpedia, dict):
                        dbpedia = list(entidades_dbpedia.keys())
                        break
                    elif isinstance(entidades_dbpedia, str):
                        # Si es string, intentar evaluarlo como diccionario
                        try:
                            import ast
                            entidades_dbpedia = ast.literal_eval(entidades_dbpedia)
                            if isinstance(entidades_dbpedia, dict):
                                dbpedia = list(entidades_dbpedia.keys())
                                break
                        except:
                            # Si no se puede evaluar, tratarlo como lista de strings
                            try:
                                dbpedia = [entidades_dbpedia] if entidades_dbpedia else []
                                break
                            except:
                                continue
                    elif isinstance(entidades_dbpedia, (list, tuple)):
                        dbpedia = list(entidades_dbpedia)
                        break

    # Procesar el texto - buscar diferentes nombres de columna
    text_column = ''
    for text_col in ['text_coref', 'coref_text', 'text', 'news']:
        if text_col in row and row[text_col]:
            text_column = str(row[text_col])
            break

    if not text_column:
        return resultado

    for frase in text_column.split('.'):
        if not frase.strip():
            continue
        try:
            anotacion = client.annotate(frase)
            sleep(sleep_time)
            #print(f"""anotacion {anotacion}""")

            for sentence in anotacion.sentence:
                d = devolver_pos(sentence)

                t = devolver_tripletas(sentence, frase, d, tripletgenerator)
                # Aplicar filtrado si está habilitado
                if enable_filtering and tripletmanager:
                    tripletas_filtradas = []
                    for tripleta in t:
                        try:
                            if tripletmanager.es_candidata(tripleta, d, ner, topic, dbpedia):
                                tripletas_filtradas.append(tripleta)
                        except:
                            tripletas_filtradas.append(tripleta)
                    resultado.extend(tripletas_filtradas)
                else:
                    resultado.extend(t)
        except:
            continue

    return resultado


def transformar_partition(df, client, sleep_time, enable_filtering, filter_method):
    """
    Transforma una partición del dataframe aplicando anotar_row a cada fila
    """
    return df.apply(
        lambda row: anotar_row(row, client, sleep_time, enable_filtering, filter_method),
        axis=1
    )


# ============================================================================
# FUNCIONES PRINCIPALES
# ============================================================================

def main():
    # ---------------- ARGUMENTOS DE LÍNEA DE COMANDOS ----------------
    parser = argparse.ArgumentParser(description='Generación de Tripletas desde Dataset')
    parser.add_argument('--input_data', type=str,
                        default='data/lda_eval/amazon/df_topic.pkl',
                        help='Ruta al archivo de datos de entrada')
    parser.add_argument('--output_dir', type=str,
                        default='data/triples_raw/amazon',
                        help='Directorio de salida para los archivos generados')
    parser.add_argument('--output_name', type=str,
                        default='amazon_semantic_tripletas_simplificado-contopic',
                        help='Nombre base para los archivos de salida')
    parser.add_argument('--corenlp_endpoint', type=str,
                        default='http://0.0.0.0:9000',
                        help='Endpoint del servidor CoreNLP')
    parser.add_argument('--npartitions', type=int,
                        default=10,
                        help='Número de particiones para Dask')
    parser.add_argument('--sleep_time', type=float,
                        default=0.30,
                        help='Tiempo de espera entre anotaciones (segundos)')
    parser.add_argument('--mode', type=str,
                        choices=['triplets_only', 'triplets_with_topics', 'add_topics_to_existing', 'explode_triplets'],
                        default='triplets_with_topics',
                        help='Modo de operación: 1=solo tripletas, 2=tripletas+tópicos, 3=añadir tópicos a tripletas existentes, 4=generar archivo de tripletas')
    parser.add_argument('--existing_triplets', type=str,
                        default='data/triples_raw/amazon/dataset_triplet_amazon_new_simplificado.csv',

                        help='Archivo de tripletas existentes (para modo add_topics_to_existing)')
    parser.add_argument('--enable_filtering', action='store_true',
                        default=False,
                        help='Habilitar filtrado de vocabulario usando entidades DBpedia y NER (por defecto: False)')
    parser.add_argument('--filter_method', type=str,
                        choices=['dbpedia_only', 'ner_only', 'both'],
                        default='both',
                        help='Método de filtrado cuando está habilitado: dbpedia_only, ner_only, o both (por defecto: both)')

    parser.add_argument('--debug_rechazadas', action='store_true',
                        default=False,
                        help='Mostrar explicaciones detalladas de por qué se rechazan tripletas')

    # Parsear argumentos con manejo de errores
    try:
        args = parser.parse_args()
    except SystemExit:
        # Si no hay argumentos o hay error, usar valores por defecto
        print("No se proporcionaron argumentos válidos. Usando valores por defecto:")
        args = argparse.Namespace(
            input_data='data/lda_eval/amazon/df_topic.pkl',
            output_dir='data/triples_raw/amazon',
            output_name='amazon_semantic_tripletas_simplificado-contopic',
            corenlp_endpoint='http://0.0.0.0:9000',
            npartitions=10,
            sleep_time=0.30,
            mode='triplets_with_topics',
            existing_triplets=None,
            enable_filtering=False,
            filter_method='both'
        )

    # ---------------- CONFIGURACIÓN ----------------
    INPUT_DATA = args.input_data
    OUTPUT_DIR = args.output_dir
    OUTPUT_NAME = args.output_name
    CORENLP_ENDPOINT = args.corenlp_endpoint
    N_PARTITIONS = args.npartitions
    SLEEP_TIME = args.sleep_time
    MODE = args.mode
    EXISTING_TRIPLETS = args.existing_triplets
    print(EXISTING_TRIPLETS)
    ENABLE_FILTERING = args.enable_filtering
    FILTER_METHOD = args.filter_method

    # Mostrar configuración
    print("=" * 60)
    print("CONFIGURACIÓN DE GENERACIÓN DE TRIPLETAS")
    print("=" * 60)
    print(f"Modo de operación: {MODE}")
    print(f"Archivo de entrada: {INPUT_DATA}")
    print(f"Directorio de salida: {OUTPUT_DIR}")
    print(f"Nombre de salida: {OUTPUT_NAME}")

    if MODE == 'triplets_with_topics':
        print(f"Archivo de tripletas existentes: {EXISTING_TRIPLETS}")
    else:
        print(f"Endpoint CoreNLP: {CORENLP_ENDPOINT}")
        print(f"Particiones Dask: {N_PARTITIONS}")
        print(f"Tiempo de espera: {SLEEP_TIME}s")
        print(f"Filtrado habilitado: {'Sí' if ENABLE_FILTERING else 'No'}")
        if ENABLE_FILTERING:
            print(f"Método de filtrado: {FILTER_METHOD}")
    print("=" * 60)

    # Crear directorio de salida si no existe
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Ejecutar según el modo seleccionado
    if MODE == 'triplets_only':
        generate_triplets_only(INPUT_DATA, OUTPUT_DIR, OUTPUT_NAME, CORENLP_ENDPOINT,
                               N_PARTITIONS, SLEEP_TIME, ENABLE_FILTERING, FILTER_METHOD)
    elif MODE == 'triplets_with_topics':
        generate_triplets_with_topics(INPUT_DATA, OUTPUT_DIR, OUTPUT_NAME,
                                      CORENLP_ENDPOINT, N_PARTITIONS, SLEEP_TIME, ENABLE_FILTERING, FILTER_METHOD)
    elif MODE == 'add_topics_to_existing':
        print(EXISTING_TRIPLETS)
        add_topics_to_existing_triplets(INPUT_DATA, OUTPUT_DIR, OUTPUT_NAME,EXISTING_TRIPLETS)
        input_data, topics_file, output_dir, output_name, existing_triplets
    else:

        explode_triplets(OUTPUT_DIR, OUTPUT_NAME,)

def generate_triplets_only(input_data, output_dir, output_name, corenlp_endpoint,
                           n_partitions, sleep_time, enable_pos_filtering, debug):
    """
    Modo 1: Generar solo tripletas sin información de tópicos
    """
    print("\n" + "=" * 50)
    print("MODO 1: GENERACIÓN SOLO DE TRIPLETAS")
    print("=" * 50)

    # Verificar archivos de entrada
    if not os.path.exists(input_data):
        print(f"ERROR: No se encontró el archivo de datos: {input_data}")
        return

    # Inicializar CoreNLP
    print("\nInicializando cliente CoreNLP...")
    try:
        client = CoreNLPClient(
            annotators=['openie'],
            endpoint=corenlp_endpoint,
            start_server=True,
            be_quiet=True
        )
        print(f"Cliente CoreNLP inicializado")
    except Exception as e:
        print(f"ERROR: No se pudo conectar a CoreNLP en {corenlp_endpoint}")
        print(f"Error: {e}")
        return

    # Cargar datos
    print(f"\nCargando datos desde: {input_data}")
    try:
        df = joblib.load(input_data)
        print(f"Datos cargados: {len(df)} filas")
        ddf = dd.from_pandas(df, npartitions=n_partitions)
    except Exception as e:
        print(f"ERROR cargando datos: {e}")
        return

    # Procesamiento principal
    print(f"\nIniciando generación de tripletas con {n_partitions} particiones...")

    try:
        # Usar map_partitions con los parámetros necesarios
        ddf['tripletas'] = ddf.map_partitions(
            transformar_partition_optimizado,
            client=client,
            topics=None,  # No hay tópicos en este modo
            sleep_time=sleep_time,
            enable_pos_filtering=enable_pos_filtering,
            debug_rechazadas = debug,
            meta=('object')
        ).compute()
        print("Generación de tripletas completado!")
    except Exception as e:
        print(f"ERROR durante el procesamiento: {e}")
        traceback.print_exc()
        return

    # Guardar resultados
    output_path = os.path.join(output_dir, output_name + "_only_triplets")
    print(f"\nGuardando resultados en: {output_path}")

    try:
        df_result = ddf.compute()
        joblib.dump(df_result, output_path)
        print(f"Datos guardados correctamente en: {output_path}")
    except Exception as e:
        print(f"ERROR guardando datos: {e}")
        return

    # Crear CSV de tripletas sin tópicos
    create_triplets_csv(df_result, output_dir, "dataset_triplet_amazon_only_triplets.csv", include_topics=False)

    print(f"\nArchivos generados:")
    print(f"- Datos con tripletas: {output_path}")
    print(f"- CSV de tripletas: {os.path.join(output_dir, 'dataset_triplet_amazon_only_triplets.csv')}")


def generate_triplets_with_topics(input_data, output_dir, output_name,
                                  corenlp_endpoint, n_partitions, sleep_time, enable_filtering, filter_method):
    """
    Modo 2: Generar tripletas con información de tópicos (comportamiento original)
    """
    print("\n" + "=" * 50)
    print("MODO 2: GENERACIÓN DE TRIPLETAS CON TÓPICOS")
    print("=" * 50)

    # Verificar archivos de entrada
    if not os.path.exists(input_data):
        print(f"ERROR: No se encontró el archivo de datos: {input_data}")
        return

    # Inicializar CoreNLP
    print("\nInicializando cliente CoreNLP...")
    try:
        client = CoreNLPClient(
            annotators=['openie'],
            endpoint=corenlp_endpoint,
            start_server=False,
            be_quiet=True
        )
        print(f"Cliente CoreNLP inicializado")
    except Exception as e:
        print(f"ERROR: No se pudo conectar a CoreNLP en {corenlp_endpoint}")
        print(f"Error: {e}")
        return

    # Cargar datos y tópicos
    print(f"\nCargando datos desde: {input_data}")
    try:
        df = joblib.load(input_data)
        print(f"Datos cargados: {len(df)} filas")
        print(f"Las columnas son {list(df.columns)}")
        ddf = dd.from_pandas(df, npartitions=n_partitions)
    except Exception as e:
        print(f"ERROR cargando datos: {e}")
        return

    # Procesamiento principal
    print(f"\nIniciando generación de tripletas con tópicos usando {n_partitions} particiones...")

    try:
        # Usar map_partitions con los parámetros necesarios
        ddf['tripletas'] = ddf.map_partitions(
            transformar_partition_optimizado,
            client=client,
           #topics=topics,
            sleep_time=sleep_time,
            debug_rechazadas=True,  # <-- Nuevo parámetro

            enable_pos_filtering=True,
            meta=('object')
        ).compute()
        print("Generación de tripletas con tópicos completado!")
    except Exception as e:
        print(f"ERROR durante el procesamiento: {e}")
        traceback.print_exc()
        return

    # Guardar resultados
    output_path = os.path.join(output_dir, output_name)
    print(f"\nGuardando resultados en: {output_path}")

    try:
        df_result = ddf.compute()
        joblib.dump(df_result, output_path)
        print(f"Datos guardados correctamente en: {output_path}")
    except Exception as e:
        print(f"ERROR guardando datos: {e}")
        return

    print(df_result.shape)
    # Crear CSV de tripletas con tópicos
    create_triplets_csv(df_result, output_dir, "dataset_triplet_amazon_new_simplificado.csv", include_topics=True)

    print(f"\nArchivos generados:")
    print(f"- Datos con tripletas: {output_path}")
    print(f"- CSV de tripletas: {os.path.join(output_dir, 'dataset_triplet_amazon_new_simplificado.csv')}")


def add_topics_to_existing_triplets(input_data, topics_file, output_dir, output_name, existing_triplets):
    """
    Modo 3: Añadir información de tópicos a tripletas ya existentes
    """
    print("\n" + "=" * 50)
    print("MODO 3: AÑADIR TÓPICOS A TRIPLETAS EXISTENTES")
    print("=" * 50)

    # Verificar archivos de entrada
    if not os.path.exists(input_data):
        print(f"ERROR: No se encontró el archivo de datos: {input_data}")
        return

    if not os.path.exists(topics_file):
        print(f"ERROR: No se encontró el archivo de tópicos: {topics_file}")
        return

    if not existing_triplets or not os.path.exists(existing_triplets):
        print(f"ERROR: No se encontró el archivo de tripletas existentes: {existing_triplets}")
        return

    # Cargar datos originales con nueva información de tópicos
    print(f"\nCargando datos actualizados desde: {input_data}")
    try:
        df_updated = joblib.load(input_data)
        print(f"Datos actualizados cargados: {len(df_updated)} filas")
    except Exception as e:
        print(f"ERROR cargando datos actualizados: {e}")
        return

    # Cargar nuevos tópicos
    print(f"\nCargando nuevos tópicos desde: {topics_file}")
    try:
        topics = joblib.load(topics_file)
        print(f"Nuevos tópicos cargados correctamente")
    except Exception as e:
        print(f"ERROR cargando tópicos: {e}")
        return

    # Cargar tripletas existentes
    print(f"\nCargando tripletas existentes desde: {existing_triplets}")
    try:
        df_existing_triplets = pd.read_csv(existing_triplets)
        print(f"Tripletas existentes cargadas: {len(df_existing_triplets)} filas")
    except Exception as e:
        print(f"ERROR cargando tripletas existentes: {e}")
        return

    # Verificar que las tripletas existentes tienen la columna necesaria
    if 'tripletas' not in df_existing_triplets.columns:
        print("ERROR: El archivo de tripletas existentes no contiene la columna 'tripletas'")
        return

    # Crear mapeo de índices a nuevos tópicos
    print("\nCreando mapeo de tópicos...")
    if 'topic' in df_updated.columns:
        topic_mapping = dict(zip(df_updated.index, df_updated['topic']))
    else:
        print("ERROR: Los datos actualizados no contienen la columna 'topic'")
        return

    # Actualizar información de tópicos en las tripletas existentes
    print("Actualizando información de tópicos...")
    df_result = df_existing_triplets.copy()

    # Actualizar la columna de tópicos si existe, o crear una nueva
    if hasattr(df_result, 'index'):
        df_result['topic'] = df_result.index.map(topic_mapping)
        print("Tópicos actualizados basándose en los índices")
    else:
        print("ADVERTENCIA: No se pudo mapear automáticamente los tópicos")
        print("Se mantendrán los tópicos originales si existen")

    # Copiar otras columnas relevantes de los datos actualizados si es necesario
    common_columns = set(df_updated.columns) & set(df_result.columns)
    for col in common_columns:
        if col not in ['tripletas', 'topic']:  # No sobrescribir estas columnas críticas
            try:
                df_result[col] = df_updated[col]
                print(f"Columna '{col}' actualizada")
            except:
                print(f"No se pudo actualizar la columna '{col}'")

    # Guardar resultados
    output_path = os.path.join(output_dir, output_name + "_updated_topics")
    print(f"\nGuardando resultados actualizados en: {output_path}")

    try:
        joblib.dump(df_result, output_path)
        print(f"Datos con tópicos actualizados guardados correctamente")
    except Exception as e:
        print(f"ERROR guardando datos: {e}")
        return

    # Crear nuevo CSV de tripletas con tópicos actualizados
    create_triplets_csv(df_result, output_dir, "dataset_triplet_amazon_updated_topics.csv", include_topics=True)

    print(f"\nArchivos generados:")
    print(f"- Datos con tópicos actualizados: {output_path}")
    print(f"- CSV de tripletas actualizadas: {os.path.join(output_dir, 'dataset_triplet_amazon_updated_topics.csv')}")


def deserializar_tripletas(triplet_str):
    """Convierte string de tripletas a lista real"""
    if pd.isna(triplet_str) or triplet_str == '' or triplet_str == '[]':
        return []

    try:
        # Intentar eval (cuidado: solo usar con datos confiables)
        import ast
        return ast.literal_eval(triplet_str)
    except:
        try:
            # Alternativa con eval si ast falla
            return eval(triplet_str)
        except:
            print(f"ADVERTENCIA: No se pudo deserializar: {triplet_str[:100]}...")
            return []


def create_triplets_csv(df, output_dir, csv_filename, include_topics=True):
    """
    Función auxiliar para crear CSV de tripletas
    """
    print(f"\nTransformando a dataframe de tripletas...")

    try:
        # Eliminar filas sin tripletas
        original_len = len(df)
        df_filtered = df[df['tripletas'].map(len) > 0].copy()
        filtered_len = len(df_filtered)
        print(f"Filas filtradas: {original_len} -> {filtered_len} ({original_len - filtered_len} sin tripletas)")

        if filtered_len == 0:
            print("ADVERTENCIA: No hay tripletas para procesar")
            return

        # Explotar tripletas

        #triplet_df_def = df_filtered.explode('tripletas')

        #triplet_df_def = triplet_df_def.reset_index()

        df_filtered['tripletas'] = df_filtered['tripletas'].apply(deserializar_tripletas)

        triplet_df_def = df_filtered.explode('tripletas', ignore_index=True)

        # Filtrar posibles None resultantes de la explosión
        triplet_df_def = triplet_df_def[triplet_df_def['tripletas'].notna()]
        triplet_df_kge = pd.DataFrame(triplet_df_def['tripletas'].tolist(), index=triplet_df_def.index)

        # Añadir metadatos - buscar diferentes nombres de columna para tópicos
        topic_col_found = None
        if include_topics:
            for topic_col in ['topic', 'new_target', 'target']:
                if topic_col in triplet_df_def.columns:
                    triplet_df_kge['new_topic'] = triplet_df_def[topic_col]
                    topic_col_found = topic_col
                    break

        triplet_df_kge['old_index'] = triplet_df_def.index

        # Configurar nombres de columnas
        if include_topics and topic_col_found:
            triplet_df_kge.columns = ['subject', 'relation', 'object', 'new_topic', 'old_index']
        else:
            triplet_df_kge.columns = ['subject', 'relation', 'object', 'old_index']

        # Guardar CSV
        csv_path = os.path.join(output_dir, csv_filename)
        triplet_df_kge.to_csv(csv_path, index=False)

        print(f"Transformación completada!")
        print(f"Total de tripletas generadas: {len(triplet_df_kge)}")
        print(f"Archivo CSV guardado en: {csv_path}")

    except Exception as e:
        print(f"ERROR durante la transformación: {e}")
        traceback.print_exc()

def explode_triplets(output_dir, output_name):
    """Función que va a generar el archivo de ternas y topics a partir de un dataframe que contiene ambas columnas"""

    output_path = os.path.join(output_dir, output_name)

    df = joblib.load(output_path)
    create_triplets_csv(df,output_dir,output_name)

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("GENERADOR DE TRIPLETAS CON MÚLTIPLES MODOS")
    print("=" * 60)
    print("Modos disponibles:")
    print("1. triplets_only: Generar solo tripletas")
    print("2. triplets_with_topics: Generar tripletas con tópicos")
    print("3. add_topics_to_existing: Añadir tópicos a tripletas existentes")
    print("")
    print("Opciones de filtrado:")
    print("--enable_filtering: Habilita filtrado por entidades (por defecto: deshabilitado)")
    print("--filter_method: Método de filtrado (dbpedia_only, ner_only, both)")
    print("=" * 60)

    main()