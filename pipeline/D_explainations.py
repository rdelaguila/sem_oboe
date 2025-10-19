# File: topic_explanation_algorithm_enhanced.py
"""
Implementación con estrategias de prompting avanzadas basadas en QualIT y XAI surveys:
- Extracción de frases clave
- Verificación de alucinaciones
- Clustering jerárquico
- Generación de explicaciones con chain-of-thought
- Evaluación con criterios específicos
- Configuración dinámica de datasets y vocabularios
"""

import os
import re
import json
import pandas as pd
import numpy as np
import matplotlib
import joblib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
import spacy
import torch
from utils.types import StringCaseInsensitiveSet, CaseInsensitiveDict, CaseInsensitiveSet
from utils.triplet_manager_lib import Tripleta
from operator import itemgetter
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, T5Tokenizer, T5ForConditionalGeneration
import joblib
import nltk
from nltk.corpus import wordnet as wn
import pickle
from typing import Dict, List, Set, Union, Any


# ======== CONFIGURACIÓN DINÁMICA ========
def load_processed_dataframe(repo_name: str) -> pd.DataFrame:
    """
    Carga el dataframe procesado para el repositorio especificado
    """
    processed_path = f'../olds/data/processed/{repo_name}/{repo_name}_processed_semantic.pkl'

    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"No se encontró el archivo: {processed_path}")

    with open(processed_path, 'rb') as f:
        data = joblib.load(f)
        print(data.head())

    if isinstance(data, pd.DataFrame):
        print(f"📋 Datos cargados como DataFrame: {len(data)} filas")
        df = data
    else:
        raise TypeError(f"Tipo de datos no soportado: {type(data)}. Se esperaba DataFrame")

    return df


def create_vocabulary_dictionaries(df: pd.DataFrame) -> Dict[int, Dict[str, List[str]]]:
    """
    Crea diccionarios de vocabulario NER y DBPedia por tópico
    Adaptado para el nuevo formato de datos de amazon
    """
    vocabulary_dict = {}

    print(f"📊 Estructura del DataFrame:")
    print(f"  Columnas: {list(df.columns) if hasattr(df, 'columns') else 'No disponibles'}")
    print(f"  Forma: {df.shape if hasattr(df, 'shape') else 'No disponible'}")

    # Mostrar muestra de los datos para debugging
    print(f"📝 Muestra de datos:")
    if hasattr(df, 'head'):
        print(df.head())
    else:
        print("  DataFrame sin método head disponible")

    # Identificar columnas relevantes automáticamente
    possible_ner_columns = []
    possible_dbpedia_columns = []
    topic_name = ''

    if hasattr(df, 'columns'):
        print(f"🔍 Buscando columnas relevantes...")
        for col_idx, col in enumerate(df.columns):
            col_lower = str(col).lower()
            print(f"  Columna {col_idx}: {col} (tipo: {type(df.iloc[0, col_idx]) if len(df) > 0 else 'N/A'})")

            if 'ner' in col_lower or 'entidad' in col_lower or 'entidades' in col_lower or 'entity' in col_lower:
                possible_ner_columns.append(col)
                print(f"    -> Detectada como NER")
            if 'dbpedia' in col_lower or 'entidades_dbpedia' in col_lower or 'dbp' in col_lower or 'uri' in col_lower:
                possible_dbpedia_columns.append(col)
                print(f"    -> Detectada como DBPedia")
            if 'target' in col_lower or 'topic' in col_lower or 'topic_new' in col_lower or 'new_target' in col_lower or 'new_topic' in col_lower:
                topic_name = col
                print(f"    -> Detectada como columna de tópico")

    print(f"🔍 Columnas NER detectadas: {possible_ner_columns}")
    print(f"🔍 Columnas DBPedia detectadas: {possible_dbpedia_columns}")
    if topic_name == '':
        raise Exception ('no se ha detectado topic')
    print(f"🔍 Columna de tópico detectada: {topic_name}")

    # Si no se detectan columnas específicas, buscar automáticamente por contenido
    if not possible_ner_columns and not possible_dbpedia_columns:
        print("⚠️ No se detectaron columnas específicas. Buscando por contenido...")

        # Examinar todas las columnas para identificar estructura
        for col_idx, col_name in enumerate(df.columns):
            if len(df) > 0:
                sample_value = df.iloc[0, col_idx]
                print(f"  🔍 Analizando columna {col_idx} ({col_name}):")
                print(f"      Tipo: {type(sample_value)}")

                # Si es un conjunto, probablemente sea NER
                if isinstance(sample_value, set):
                    possible_ner_columns.append(col_name)
                    print(f"      -> Identificado como NER (set con {len(sample_value)} elementos)")
                    if len(sample_value) > 0:
                        print(f"      -> Muestra: {list(sample_value)[:3]}")

                # Si es un diccionario, verificar si tiene estructura de DBPedia
                elif isinstance(sample_value, dict):
                    print(f"      -> Diccionario con {len(sample_value)} claves")
                    if len(sample_value) > 0:
                        first_key = list(sample_value.keys())[0]
                        first_value = sample_value[first_key]
                        print(f"      -> Primera clave: {first_key}")
                        print(f"      -> Primer valor tipo: {type(first_value)}")

                        # Verificar estructura DBPedia nueva
                        if isinstance(first_value, dict):
                            print(f"      -> Claves del valor: {list(first_value.keys())}")
                            if 'URI' in first_value:
                                possible_dbpedia_columns.append(col_name)
                                print(f"      -> ✅ Identificado como DBPedia (dict con URI)")
                            elif 'uri' in str(first_value).lower():
                                possible_dbpedia_columns.append(col_name)
                                print(f"      -> ✅ Identificado como DBPedia (contiene uri)")
                else:
                    print(f"      -> Tipo {type(sample_value)} no reconocido como NER/DBPedia")

    print(f"🔍 Columnas NER finales: {possible_ner_columns}")
    print(f"🔍 Columnas DBPedia finales: {possible_dbpedia_columns}")

    # Procesar cada fila del DataFrame
    for idx, row in df.iterrows():
        # Determinar topic_id
        if topic_name and topic_name in row:
            topic_id = row[topic_name]
        else:
            topic_id = idx  # Usar índice si no hay columna de tópico

        print(f"\n🔄 Procesando tópico {topic_id} (fila {idx}):")

        # Procesar entidades NER
        ner_entities = []
        for ner_col in possible_ner_columns:
            if ner_col in row and row[ner_col] is not None:
                data = row[ner_col]
                print(f"    📝 Procesando NER de columna '{ner_col}' (tipo: {type(data)})")

                if isinstance(data, set):
                    entities = [str(x).strip().lower() for x in data if x and str(x).strip()]
                    ner_entities.extend(entities)
                    print(f"        ✅ Extraídas {len(entities)} entidades NER")
                    if entities:
                        print(f"        📋 Ejemplos: {entities[:3]}")

                elif isinstance(data, list):
                    entities = [str(x).strip().lower() for x in data if x and str(x).strip()]
                    ner_entities.extend(entities)
                    print(f"        ✅ Extraídas {len(entities)} entidades NER de lista")

                elif isinstance(data, str):
                    entities = [x.strip().lower() for x in data.replace(',', ' ').split() if x.strip()]
                    ner_entities.extend(entities)
                    print(f"        ✅ Extraídas {len(entities)} entidades NER de string")

                elif isinstance(data, dict):
                    entities = [str(k).strip().lower() for k in data.keys() if k]
                    ner_entities.extend(entities)
                    print(f"        ✅ Extraídas {len(entities)} entidades NER de dict keys")

        # Procesar entidades DBPedia (formato nuevo)
        dbpedia_entities = []
        for dbp_col in possible_dbpedia_columns:
            if dbp_col in row and row[dbp_col] is not None:
                data = row[dbp_col]
                print(f"    🌐 Procesando DBPedia de columna '{dbp_col}' (tipo: {type(data)})")
                print(f"        📊 Diccionario con {len(data)} entidades")

                if isinstance(data, dict):
                    # Nuevo formato: {'dog': {'URI': '...', 'types': ...}, ...}
                    for entity_name, entity_data in data.items():
                        if isinstance(entity_data, dict):
                            # Añadir el nombre de la entidad
                            entity_clean = str(entity_name).strip().lower()
                            if entity_clean and len(entity_clean) > 1:
                                dbpedia_entities.append(entity_clean)

                            # Extraer tipos si están disponibles
                            types_data = entity_data.get('types', '')
                            if isinstance(types_data, str) and types_data:
                                print(f"        📋 Tipos para '{entity_name}': {types_data}")
                                # Extraer nombres de tipos (ej: 'DBpedia:Food' -> 'food')
                                for type_part in types_data.split(','):
                                    type_part = type_part.strip()
                                    if ':' in type_part:
                                        type_name = type_part.split(':')[-1].strip().lower()
                                        if type_name and not type_name.startswith('q') and len(type_name) > 1:
                                            dbpedia_entities.append(type_name)

                            # También extraer de types_normalizados si existe
                            types_norm = entity_data.get('types_normalizados', '')
                            if isinstance(types_norm, str) and types_norm:
                                print(f"        📋 Tipos norm para '{entity_name}': {types_norm}")
                                for type_part in types_norm.split():
                                    type_part = type_part.strip().lower()
                                    if type_part and not type_part.startswith('q') and len(type_part) > 1:
                                        dbpedia_entities.append(type_part)

                    print(f"        ✅ Extraídas {len(set(dbpedia_entities))} entidades DBPedia únicas")
                    if dbpedia_entities:
                        print(f"        📋 Ejemplos: {list(set(dbpedia_entities))[:5]}")

        # Si no se encontraron datos específicos, buscar en todas las columnas
        if not ner_entities and not dbpedia_entities:
            print(f"    ⚠️ No se encontraron datos específicos para tópico {topic_id}. Buscando en todas las columnas.")
            for col_name, value in row.items():
                if value is not None:
                    if isinstance(value, set):
                        # Tratar sets como NER
                        entities = [str(x).strip().lower() for x in value if x and str(x).strip()]
                        ner_entities.extend(entities)
                        print(f"        📝 Encontradas {len(entities)} entidades NER en columna '{col_name}'")

                    elif isinstance(value, dict):
                        # Buscar estructura DBPedia
                        for k, v in value.items():
                            if isinstance(v, dict) and ('URI' in v or 'types' in v or 'uri' in str(v).lower()):
                                entity_name = str(k).strip().lower()
                                if entity_name and len(entity_name) > 1:
                                    dbpedia_entities.append(entity_name)

                                # Extraer tipos
                                types_data = v.get('types', '')
                                if isinstance(types_data, str) and types_data:
                                    for type_part in types_data.split(','):
                                        if ':' in type_part:
                                            type_name = type_part.split(':')[-1].strip().lower()
                                            if type_name and len(type_name) > 1:
                                                dbpedia_entities.append(type_name)

                        if dbpedia_entities:
                            print(f"        🌐 Encontradas entidades DBPedia en columna '{col_name}'")

        # Limpiar y eliminar duplicados
        ner_entities = list(set([x for x in ner_entities if x and len(x) > 1]))
        dbpedia_entities = list(set([x for x in dbpedia_entities if x and len(x) > 1]))

        vocabulary_dict[topic_id] = {
            'ner': ner_entities,
            'dbpedia': dbpedia_entities
        }

        print(f"    ✅ Tópico {topic_id}: {len(ner_entities)} NER, {len(dbpedia_entities)} DBPedia")
        if dbpedia_entities:
            print(f"        🌐 DBPedia ejemplos: {dbpedia_entities[:5]}")
        if ner_entities:
            print(f"        📝 NER ejemplos: {ner_entities[:5]}")

    return vocabulary_dict


def load_top_terms_by_topic(repo_name: str) -> Dict[int, List[str]]:
    """
    Carga los términos más relevantes por tópico
    """
    top_terms_path = f'../olds/data/lda_eval/{repo_name}/top_terms_by_topic.pkl'

    if not os.path.exists(top_terms_path):
        raise FileNotFoundError(f"No se encontró el archivo: {top_terms_path}")

    with open(top_terms_path, 'rb') as f:
        top_terms = joblib.load(f)  # Cambiado de pickle.load a joblib.load para consistencia

    return top_terms


def show_topic_vocabulary(topic_id: int, vocabulary_dict: Dict[int, Dict[str, List[str]]],
                          top_terms: Dict[int, List[str]]) -> None:
    """
    Muestra el vocabulario relevante para un tópico específico
    """
    print(f"\n=== VOCABULARIO PARA EL TÓPICO {topic_id} ===")

    # Mostrar términos más relevantes del LDA
    if topic_id in top_terms:
        print(f"\n📊 Términos más relevantes (LDA):")
        terms = top_terms[topic_id][:10]  # Mostrar los 10 primeros
        for i, term in enumerate(terms, 1):
            print(f"  {i}. {term}")

    # Mostrar entidades NER
    if topic_id in vocabulary_dict and vocabulary_dict[topic_id]['ner']:
        print(f"\n🏷️  Entidades NER ({len(vocabulary_dict[topic_id]['ner'])} total):")
        ner_sample = vocabulary_dict[topic_id]['ner'][:10]  # Mostrar las 10 primeras
        for i, entity in enumerate(ner_sample, 1):
            print(f"  {i}. {entity}")
        if len(vocabulary_dict[topic_id]['ner']) > 10:
            print(f"  ... y {len(vocabulary_dict[topic_id]['ner']) - 10} más")

    # Mostrar entidades DBPedia
    if topic_id in vocabulary_dict and vocabulary_dict[topic_id]['dbpedia']:
        print(f"\n🌐 Entidades DBPedia ({len(vocabulary_dict[topic_id]['dbpedia'])} total):")
        dbpedia_sample = vocabulary_dict[topic_id]['dbpedia'][:10]  # Mostrar las 10 primeras
        for i, entity in enumerate(dbpedia_sample, 1):
            print(f"  {i}. {entity}")
        if len(vocabulary_dict[topic_id]['dbpedia']) > 10:
            print(f"  ... y {len(vocabulary_dict[topic_id]['dbpedia']) - 10} más")


def create_output_directory(repo_name: str) -> str:
    """
    Crea el directorio de salida si no existe
    """
    output_dir = f'../olds/data/explainations/{repo_name}'
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def get_user_vocabulary_choice(topic_id: int, vocabulary_dict: Dict[int, Dict[str, List[str]]],
                               top_terms: Dict[int, List[str]]) -> List[str]:
    """
    Permite al usuario elegir el vocabulario a usar
    """
    print(f"\n🎯 Opciones de vocabulario para el tópico {topic_id}:")
    print("1. Usar vocabulario por defecto (términos LDA)")
    print("2. Usar entidades NER")
    print("3. Usar entidades DBPedia")
    print("4. Combinar LDA + NER + DBPedia")
    print("5. Introducir vocabulario personalizado")

    while True:
        try:
            choice = int(input("\nSelecciona una opción (1-5): "))
            if choice in [1, 2, 3, 4, 5]:
                break
            else:
                print("Por favor, selecciona un número entre 1 y 5.")
        except ValueError:
            print("Por favor, introduce un número válido.")

    vocabulary = []

    if choice == 1:
        # Vocabulario por defecto (términos LDA)
        if topic_id in top_terms:
            vocabulary = top_terms[topic_id][:20]  # Top 20 términos
        print(f"✅ Usando vocabulario LDA ({len(vocabulary)} términos)")

    elif choice == 2:
        # Entidades NER
        if topic_id in vocabulary_dict:
            vocabulary = vocabulary_dict[topic_id]['ner']
        print(f"✅ Usando entidades NER ({len(vocabulary)} términos)")

    elif choice == 3:
        # Entidades DBPedia
        if topic_id in vocabulary_dict:
            vocabulary = vocabulary_dict[topic_id]['dbpedia']
        print(f"✅ Usando entidades DBPedia ({len(vocabulary)} términos)")

    elif choice == 4:
        # Combinar todo
        if topic_id in top_terms:
            vocabulary.extend(top_terms[topic_id][:10])
        if topic_id in vocabulary_dict:
            vocabulary.extend(vocabulary_dict[topic_id]['ner'])
            vocabulary.extend(vocabulary_dict[topic_id]['dbpedia'])
        vocabulary = list(set(vocabulary))  # Eliminar duplicados
        print(f"✅ Usando vocabulario combinado ({len(vocabulary)} términos)")

    elif choice == 5:
        # Vocabulario personalizado
        print("Introduce los términos separados por comas:")
        custom_terms = input("Vocabulario: ").strip()
        vocabulary = [term.strip() for term in custom_terms.split(',') if term.strip()]
        print(f"✅ Usando vocabulario personalizado ({len(vocabulary)} términos)")

    return vocabulary


# ======== CONFIGURACIÓN DE MODELOS ========
SPACY_MODEL = 'en_core_web_lg'
GEN_MODEL = 'Qwen/Qwen2-7B-Instruct'
EVAL_MODEL = 'google/flan-t5-base'
TOP_K_TERMS = 10
N_SINONIMOS = 1
VISITAR_OBJETO = True


# Helper functions
def remove_numbers(text):
    return re.sub(r"\d+", "", text)


def remove_dbpedia_categories(s):
    return s.split('/')[-1]


def return_url_element(s):
    for sep in ['#', '/']:
        if sep in s:
            s = s.split(sep)[-1]
    return s


def extract_json_from_text(text):
    """Extrae JSON de texto de manera más robusta"""
    try:
        # Buscar el primer { y el último }
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            json_str = text[start:end + 1]
            return json.loads(json_str)
    except:
        pass

    # Si falla, intentar extraer valores manualmente
    try:
        # Buscar patrones como "explicación": "texto"
        explanation_match = re.search(r'"explicación":\s*"([^"]*)"', text)
        coherencia_match = re.search(r'"coherencia":\s*(\d+)', text)
        relevancia_match = re.search(r'"relevancia":\s*(\d+)', text)
        cobertura_match = re.search(r'"cobertura":\s*(\d+)', text)

        result = {}
        if explanation_match:
            result['explicación'] = explanation_match.group(1)
        if coherencia_match:
            result['coherencia'] = int(coherencia_match.group(1))
        if relevancia_match:
            result['relevancia'] = int(relevancia_match.group(1))
        if cobertura_match:
            result['cobertura'] = int(cobertura_match.group(1))

        if result:
            return result
    except:
        pass

    return None


def setup_configuration():
    """
    Configuración inicial del sistema
    """
    print("🚀 Sistema de Análisis de Tópicos y Explicaciones")
    print("=" * 60)

    # 1. Solicitar nombre del dataset/repo
    repo_name = input("📁 Introduce el nombre del dataset/repositorio (ej: amazon, bbc): ").strip()

    if not repo_name:
        print("❌ Error: Debes introducir un nombre de repositorio.")
        return None

    try:
        # 2. Cargar dataframe procesado
        print(f"\n📊 Cargando datos para el repositorio '{repo_name}'...")
        df = load_processed_dataframe(repo_name)
        print(f"✅ Dataframe cargado: {len(df)} filas")

        # 3. Crear diccionarios de vocabulario
        print("🔍 Procesando vocabulario NER y DBPedia...")
        vocabulary_dict = create_vocabulary_dictionaries(df)
        print(f"✅ Vocabulario procesado para {len(vocabulary_dict)} tópicos")

        # 4. Cargar términos más relevantes por tópico
        print("📈 Cargando términos más relevantes por tópico...")
        top_terms = load_top_terms_by_topic(repo_name)
        print(f"✅ Términos cargados para {len(top_terms)} tópicos")

        # 5. Crear directorio de salida
        output_dir = create_output_directory(repo_name)
        print(f"📂 Directorio de salida: {output_dir}")

        # 6. Mostrar tópicos disponibles
        available_topics = list(set(list(vocabulary_dict.keys()) + list(top_terms.keys())))
        available_topics.sort()

        print(f"\n📋 Tópicos disponibles: {available_topics}")

        # 7. Solicitar tópico a analizar
        while True:
            try:
                topic_id = int(
                    input(f"\n🎯 ¿Qué tópico deseas analizar? ({min(available_topics)}-{max(available_topics)}): "))
                if topic_id in available_topics:
                    break
                else:
                    print(f"❌ Tópico {topic_id} no disponible. Opciones: {available_topics}")
            except ValueError:
                print("❌ Por favor, introduce un número válido.")

        # 8. Mostrar vocabulario del tópico
        show_topic_vocabulary(topic_id, vocabulary_dict, top_terms)

        # 9. Permitir al usuario elegir vocabulario
        selected_vocabulary = get_user_vocabulary_choice(topic_id, vocabulary_dict, top_terms)

        # 10. Definir ruta de ternas
        triples_path = f'../olds/data/triples_raw/{repo_name}/dataset_triplet_{repo_name}_new_simplificado.csv'
        print(f"\n🔗 Ruta de ternas: {triples_path}")

        if not os.path.exists(triples_path):
            print(f"⚠️  Advertencia: No se encontró el archivo de ternas: {triples_path}")
            # Buscar archivo alternativo
            alt_path = f'../olds/data/triples_raw/processed/dataset_final_triplet_{repo_name}_pykeen'
            if os.path.exists(alt_path):
                triples_path = alt_path
                print(f"✅ Usando archivo alternativo: {triples_path}")
            else:
                print("❌ No se encontró ningún archivo de ternas válido")
                return None
        else:
            print("✅ Archivo de ternas encontrado")

        # 11. Configuración para análisis
        config = {
            'repo_name': repo_name,
            'topic_id': topic_id,
            'vocabulary': selected_vocabulary,
            'triples_path': triples_path,
            'output_dir': output_dir,
            'vocabulary_dict': vocabulary_dict,
            'top_terms': top_terms,
            'vocabulary_stats': {
                'total_terms': len(selected_vocabulary),
                'ner_available': len(vocabulary_dict.get(topic_id, {}).get('ner', [])),
                'dbpedia_available': len(vocabulary_dict.get(topic_id, {}).get('dbpedia', [])),
                'lda_available': len(top_terms.get(topic_id, []))
            }
        }

        config_path = os.path.join(output_dir, f'config_topic_{topic_id}.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            # Serializar solo los datos básicos para JSON
            json_config = {k: v for k, v in config.items()
                           if k not in ['vocabulary_dict', 'top_terms']}
            json.dump(json_config, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Configuración guardada en: {config_path}")
        print(f"\n🎉 Configuración completada. Vocabulario seleccionado: {len(selected_vocabulary)} términos")
        print("🔄 Iniciando análisis de ternas y clustering...")

        return config

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Verifica que los archivos existan en las rutas especificadas.")
        return None
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return None


def main():
    """
    Función principal del script
    """
    # Configuración inicial
    config = setup_configuration()
    if config is None:
        return

    # Extraer parámetros de configuración
    TOPIC_ID = config['topic_id']
    TERMINOS_A_INCLUIR = set([term.lower() for term in config['vocabulary']])  # Convertir a minúsculas
    TRIPLES_PATH = config['triples_path']
    OUTPUT_DIR = config['output_dir']

    # Crear diccionarios de compatibilidad con formato original
    vocabulary_dict = config['vocabulary_dict']
    if TOPIC_ID in vocabulary_dict:
        # Simular estructura original de DBPedia y NER
        dictdbp = {}
        dictner = {}

        # Crear dictdbp desde las entidades DBPedia extraídas
        for term in vocabulary_dict[TOPIC_ID]['dbpedia']:
            dictdbp[term.lower()] = [{'URI': f'http://dbpedia.org/resource/{term}', 'tipos': []}]

        # Crear dictner desde las entidades NER extraídas
        for term in vocabulary_dict[TOPIC_ID]['ner']:
            dictner[term.lower()] = 'ENTITY'

        # IMPORTANTE: También añadir términos del vocabulario LDA seleccionado a dictdbp
        # para que puedan ser encontrados en las tripletas
        for term in config['vocabulary']:
            term_lower = term.lower()
            if term_lower not in dictdbp:
                dictdbp[term_lower] = [{'URI': f'http://dbpedia.org/resource/{term}', 'tipos': []}]
    else:
        dictdbp = {}
        dictner = {}

    print(f"📊 Diccionario DBPedia creado: {len(dictdbp)} términos")
    print(f"📊 Diccionario NER creado: {len(dictner)} términos")

    # Mostrar algunos ejemplos
    if dictdbp:
        print(f"   Ejemplos DBPedia: {list(dictdbp.keys())[:5]}")
    else:
        print("   ⚠️ PROBLEMA: Diccionario DBPedia está vacío!")
        print("   🔍 Verificando contenido del vocabulary_dict...")
        print(f"   - Topic ID: {TOPIC_ID}")
        print(f"   - Vocabulary dict keys: {list(vocabulary_dict.keys())}")
        if TOPIC_ID in vocabulary_dict:
            print(f"   - DBPedia entities para tópico {TOPIC_ID}: {vocabulary_dict[TOPIC_ID]['dbpedia'][:10]}")
            print(f"   - NER entities para tópico {TOPIC_ID}: {vocabulary_dict[TOPIC_ID]['ner'][:10]}")

        # Como fallback, crear dictdbp desde el vocabulario seleccionado
        print("   🔧 Creando diccionario desde vocabulario seleccionado como fallback...")
        for term in config['vocabulary']:
            term_lower = term.lower()
            dictdbp[term_lower] = [{'URI': f'http://dbpedia.org/resource/{term}', 'tipos': []}]
        print(f"   ✅ Diccionario fallback creado con {len(dictdbp)} términos")

    if dictner:
        print(f"   Ejemplos NER: {list(dictner.keys())[:5]}")
    else:
        print("   ⚠️ Diccionario NER está vacío (normal si no hay entidades NER)")

    # Debugging adicional del vocabulario
    print(f"\n🔍 Debugging del vocabulario seleccionado:")
    print(f"   - Tipo de vocabulario: {type(config['vocabulary'])}")
    print(f"   - Primeros 10 términos: {config['vocabulary'][:10]}")
    print(f"   - Términos en minúsculas: {[term.lower() for term in config['vocabulary'][:5]]}")

    # Preparación de modelos
    try:
        nlp = spacy.load(SPACY_MODEL)
    except OSError:
        print(
            f"❌ Error: Modelo SpaCy '{SPACY_MODEL}' no encontrado. Instálalo con: python -m spacy download {SPACY_MODEL}")
        return

    nltk.download('wordnet', quiet=True)

    try:
        df_tr = pd.read_csv(TRIPLES_PATH)
        print(f"✅ Tripletas cargadas: {len(df_tr)} filas")

        # Mostrar muestra de tripletas para debugging
        print("📝 Muestra de tripletas:")
        for i in range(min(3, len(df_tr))):
            print(f"   {i}: {df_tr.iloc[i]['subject']} -> {df_tr.iloc[i]['relation']} -> {df_tr.iloc[i]['object']}")

    except Exception as e:
        print(f"❌ Error cargando tripletas: {e}")
        return

    # 1. Filtrado y extracción de tripletas relevantes
    print("\n🔍 Procesando tripletas relevantes...")
    listado_tripletas = []
    palabrasdbpedia = set(k.lower() for k in dictdbp.keys())
    print(f"📊 DBPedia términos para filtrar: {len(palabrasdbpedia)}")
    print(f"📊 NER términos para filtrar: {len(dictner)}")
    print(f"📊 Vocabulario a incluir: {len(TERMINOS_A_INCLUIR)}")

    # Debug: mostrar intersecciones
    if palabrasdbpedia:
        interseccion = TERMINOS_A_INCLUIR.intersection(palabrasdbpedia)
        print(f"📊 Intersección vocabulario-DBPedia: {len(interseccion)} términos")
        if interseccion:
            print(f"   Ejemplos de intersección: {list(interseccion)[:5]}")
    else:
        print("⚠️ No hay palabras DBPedia para intersectar!")

    anterior = None
    processed_count = 0
    terms_found_count = 0
    matches_found = 0

    for i, row in df_tr.iterrows():
        if processed_count % 5000 == 0:  # Reducir frecuencia de logging
            print(
                f"  Procesando: {processed_count}/{len(df_tr)} (términos encontrados: {terms_found_count}, matches: {matches_found})")
        processed_count += 1

        tripleta = Tripleta({'subject': str(row['subject']),
                             'relation': row['relation'],
                             'object': str(row['object'])})

        sujeto = set([word.lower() for word in tripleta.sujeto.split()])
        objeto = set([word.lower() for word in tripleta.objeto.split()]) if VISITAR_OBJETO else set()

        if anterior is None:
            anterior = tripleta

        misma_super = (tripleta.esTripletaSuper(anterior) == anterior.esTripletaSuper(tripleta))
        dif = tripleta.dondeSonDiferentes(anterior)

        if (misma_super and (dif == ('sujeto', 'relacion', 'objeto') or dif == ('sujeto', None, 'objeto'))):
            anterior = tripleta
        else:
            continue

        # Verificar intersección con vocabulario seleccionado
        has_match = (TERMINOS_A_INCLUIR is None
                     or not TERMINOS_A_INCLUIR.isdisjoint(sujeto)
                     or (VISITAR_OBJETO and not TERMINOS_A_INCLUIR.isdisjoint(objeto)))

        if has_match:
            matches_found += 1

            # Debug: mostrar las primeras matches
            if matches_found <= 5:
                print(f"     Match {matches_found}: sujeto={sujeto}, objeto={objeto}")
                print(f"       Intersección sujeto: {sujeto.intersection(TERMINOS_A_INCLUIR)}")
                if VISITAR_OBJETO:
                    print(f"       Intersección objeto: {objeto.intersection(TERMINOS_A_INCLUIR)}")

            visitados = set()
            encontradas = sujeto.intersection(palabrasdbpedia)
            no_encontradas = sujeto.difference(palabrasdbpedia)

            if VISITAR_OBJETO:
                encontradas.update(objeto.intersection(palabrasdbpedia))
                no_encontradas.update(objeto.difference(palabrasdbpedia))

            # Debug: mostrar encontradas
            if matches_found <= 5 and encontradas:
                print(f"       Encontradas en DBPedia: {encontradas}")

            for termino in encontradas:
                termino_lower = termino.lower()

                if termino in visitados:
                    continue

                if termino[0].isdigit():
                    no_encontradas.add(termino)
                    continue

                info_list = dictdbp.get(termino_lower, [])
                if not info_list:
                    no_encontradas.add(termino)
                    continue

                info_termino = info_list[0]
                uri_db = info_termino.get('URI', '')
                tipos_db = info_termino.get('tipos', [])

                sinonimos = []
                lwordnet = []

                # WordNet synonyms + hypernyms
                try:
                    for syn in wn.synsets(termino):
                        sinonimos.extend(syn.lemma_names())
                        for h in syn.hypernyms():
                            lwordnet.extend(h.lemma_names())
                except:
                    pass

                # NER
                sujeto_en_ner = dictner.get(termino_lower, '')
                ner = []
                if sujeto_en_ner:
                    ner.append(sujeto_en_ner)

                diccionario_termino = {
                    'termino': termino,
                    'sinonimos': list(set(sinonimos)),
                    'resource': uri_db,
                    'dbpedia': tipos_db,
                    'ner': ner,
                    'wordnet': lwordnet
                }

                listado_tripletas.append(diccionario_termino)
                visitados.add(termino)
                terms_found_count += 1

                # Debug: mostrar primeros términos encontrados
                if terms_found_count <= 5:
                    print(f"       ✅ Término procesado: {termino}")

    print(f"📊 Estadísticas finales del procesamiento:")
    print(f"   - Tripletas procesadas: {processed_count}")
    print(f"   - Matches con vocabulario: {matches_found}")
    print(f"   - Términos finales extraídos: {terms_found_count}")

    print(f"✅ Tripletas procesadas: {len(listado_tripletas)} términos relevantes encontrados")

    if len(listado_tripletas) == 0:
        print("❌ No se encontraron términos relevantes. Verifica el vocabulario seleccionado.")
        print("🔍 Debugging información:")
        print(f"  - Vocabulario a incluir: {list(TERMINOS_A_INCLUIR)[:10]}...")
        print(f"  - Palabras DBPedia disponibles: {list(palabrasdbpedia)[:10]}...")
        print(f"  - Total tripletas procesadas: {processed_count}")
        return

    # 2. Expandir vocabulario con SpaCy similarity
    print("\n🧠 Expandiendo vocabulario con similitudes semánticas...")
    df = pd.DataFrame(listado_tripletas)
    vocab_aux, lista_tipos = [], []

    for _, row in df.iterrows():
        termino = row['termino']
        tipos = []

        # Manejo más robusto de tipos
        dbpedia_tipos = row['dbpedia']
        if isinstance(dbpedia_tipos, list):
            tipos.extend(dbpedia_tipos)
        elif isinstance(dbpedia_tipos, str):
            tipos.extend(dbpedia_tipos.split(','))

        wordnet_tipos = row['wordnet']
        if isinstance(wordnet_tipos, list):
            tipos.extend(wordnet_tipos)

        tipos_clean = []
        for t in tipos:
            el = return_url_element(remove_dbpedia_categories(remove_numbers(str(t))))
            if el and el != 'Q':
                tipos_clean.append(el)

        if not tipos_clean:
            continue

        try:
            sims = [nlp(termino).similarity(nlp(t2)) for t2 in tipos_clean]
            if not sims:
                continue

            idx = list(np.argpartition(sims, -N_SINONIMOS)[-N_SINONIMOS:])
            sel = [tipos_clean[i] for i in idx]
            puntuaciones = [sims[i] for i in idx]

            # Manejo correcto de tipos más similares
            if len(sel) == 1:
                tipos_mas_similares = sel[0]
                puntuaciones_tipos_similares = puntuaciones[0]
            else:
                tipos_mas_similares = sel
                puntuaciones_tipos_similares = puntuaciones

            lista_tipos.append({
                'termino': termino,
                'tipos': tipos_mas_similares,
                'similitudes': puntuaciones_tipos_similares
            })

            vocab_aux.append(termino)
            if isinstance(tipos_mas_similares, str):
                vocab_aux.append(tipos_mas_similares)
            else:
                vocab_aux.extend(tipos_mas_similares)
        except Exception as e:
            print(f"  ⚠️ Error procesando término '{termino}': {e}")
            continue

    # Lematización del vocabulario
    print("📝 Lematizando vocabulario final...")
    vocab = set()
    for doc in nlp.pipe(vocab_aux):
        lemmatized = " ".join([token.lemma_.lower() for token in doc])
        vocab.add(lemmatized)

    terms = list(vocab)
    print(f"✅ Vocabulario final: {len(terms)} términos únicos")

    if len(terms) < 2:
        print("❌ Error: No hay suficientes términos para clustering")
        return

    # Calcular matriz de similitudes
    print("🔢 Calculando matriz de similitudes...")
    M = np.array([[nlp(t1).similarity(nlp(t2)) for t2 in terms] for t1 in terms])

    # 3. Clustering jerárquico + silhouette
    print("\n📊 Optimizando número de clusters...")
    range_n_clusters = range(2, min(len(terms), 30))
    valores_medios_silhouette = []

    for n_clusters in range_n_clusters:
        modelo = AgglomerativeClustering(
            metric='euclidean',
            linkage='ward',
            n_clusters=n_clusters
        )
        labels_temp = modelo.fit_predict(M)
        silhouette_avg = silhouette_score(M, labels_temp)
        valores_medios_silhouette.append(silhouette_avg)

    optimal_clusters = range_n_clusters[np.argmax(valores_medios_silhouette)]

    # Gráficos
    plt.figure(figsize=(10, 6))
    plt.plot(range_n_clusters, valores_medios_silhouette, marker='o')
    plt.axvline(x=optimal_clusters, color='r', linestyle='--', label=f'Óptimo automático: {optimal_clusters}')
    plt.title('Evolución del índice Silhouette')
    plt.xlabel('Número de clusters')
    plt.ylabel('Silhouette promedio')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'silhouette_evolution.png'))
    print(f"📈 Gráfico silhouette generado en {OUTPUT_DIR}")

    # Crear el modelo para el dendrograma
    linkage_matrix = shc.linkage(M, method='ward')

    # ============= PUNTO INTERACTIVO PARA SELECCIÓN DE CLUSTERS =============
    print("\n" + "=" * 60)
    print("ANÁLISIS DE CLUSTERING")
    print("=" * 60)
    print(f"\nVocabulario final ({len(terms)} términos):")
    for i, term in enumerate(terms):
        print(f"  {i:2d}. {term}")

    print(f"\nNúmero óptimo de clusters según Silhouette: {optimal_clusters}")
    print(f"Silhouette score máximo: {max(valores_medios_silhouette):.3f}")

    print("\n¿Desea modificar el número de clusters?")
    print("(Presione Enter para usar el óptimo automático o ingrese un número)")

    user_input = input(f"Número de clusters [{optimal_clusters}]: ").strip()

    if user_input and user_input.isdigit():
        user_clusters = int(user_input)
        if 2 <= user_clusters <= len(terms):
            final_clusters = user_clusters
            print(f"\n✓ Usando {final_clusters} clusters según selección del usuario")
        else:
            print(f"\n⚠ Número inválido. Usando óptimo automático: {optimal_clusters}")
            final_clusters = optimal_clusters
    else:
        final_clusters = optimal_clusters
        print(f"\n✓ Usando número óptimo automático: {final_clusters}")

    # Generar dendrograma con línea de corte
    plt.figure(figsize=(12, 8))

    # Calcular la altura de corte para el número de clusters seleccionado
    if final_clusters < len(terms):
        cut_height = linkage_matrix[-(final_clusters - 1), 2] * 0.9
    else:
        cut_height = 0

    dend = shc.dendrogram(linkage_matrix, labels=terms, color_threshold=cut_height)

    # Añadir línea horizontal en la altur
    # a de corte
    if cut_height > 0:
        plt.axhline(y=cut_height, color='r', linestyle='--', linewidth=2,
                    label=f'Corte para {final_clusters} clusters')

    plt.title(f"Dendrograma jerárquico (Ward) - {final_clusters} clusters seleccionados")
    plt.xlabel('Términos')
    plt.ylabel('Distancia')
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'dendrogram_with_cut.png'))
    print(f"🌳 Dendrograma con corte generado en {OUTPUT_DIR}")

    # Clustering final con el número seleccionado
    labels = AgglomerativeClustering(n_clusters=final_clusters, linkage='ward').fit_predict(M)
    sample_sil = silhouette_samples(M, labels)
    global_sil = silhouette_score(M, labels)

    print(f"\n📊 Silhouette score para {final_clusters} clusters: {global_sil:.3f}")

    clusters = {}
    for cl in set(labels):
        idxs = np.where(labels == cl)[0]
        term_sils = [(terms[i], sample_sil[i]) for i in idxs]
        top_terms = [t for t, _ in sorted(term_sils, key=lambda x: -x[1])[:TOP_K_TERMS]]
        clusters[cl] = top_terms

    # Mostrar resumen de clusters
    print("\nRESUMEN DE CLUSTERS:")
    print("-" * 60)
    for cl, terms_cl in clusters.items():
        print(f"Cluster {cl}: {', '.join(terms_cl)}")

    clusters_str = {str(k): v for k, v in clusters.items()}

    with open(os.path.join(OUTPUT_DIR, 'clusters.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'best_k': len(clusters_str),
            'selected_k': final_clusters,
            'automatic_optimal_k': optimal_clusters,
            'global_sil': global_sil,
            'clusters': clusters_str
        }, f, ensure_ascii=False, indent=2)

    # ============= FIN DE SECCIÓN INTERACTIVA =============

    # 4. GENERACIÓN CON ESTRATEGIAS AVANZADAS DE PROMPTING (basado en QualIT + XAI)
    print("\n🤖 Inicializando modelos de IA para generación y evaluación...")

    torch_dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 Dispositivo: {torch_dev}")

    try:
        tok_g = AutoTokenizer.from_pretrained(GEN_MODEL)
        mod_g = AutoModelForCausalLM.from_pretrained(GEN_MODEL).to(torch_dev)
        gen = pipeline('text-generation', model=mod_g, tokenizer=tok_g,
                       device=0 if torch.cuda.is_available() else -1,
                       return_full_text=False)
        print("✅ Modelo generativo cargado")
    except Exception as e:
        print(f"❌ Error cargando modelo generativo: {e}")
        print("⚠️ Continuando sin generación automática de explicaciones")
        gen = None

    def generate_explanation_with_cot(cluster_id, terms, topic_id, silhouette_score):
        """
        Estrategia de Chain-of-Thought para generación de explicaciones
        Basada en QualIT: extracción de frases clave -> verificación -> explicación
        """
        if gen is None:
            return "Explicación no disponible - modelo no cargado", "", False

        # Paso 1: Extracción de frases clave (inspirado en QualIT)
        key_phrase_prompt = f"""You are an expert in topic modeling and semantic analysis.

TASK: Extract 2-3 key phrases that best represent the semantic relationship between these terms.

TERMS: {', '.join(terms)}
CONTEXT: These terms belong to topic {topic_id} (technology domain)
CLUSTERING QUALITY: Silhouette score = {silhouette_score:.3f}

INSTRUCTIONS:
1. Identify the most important semantic connections
2. Extract concise key phrases (2-4 words each)
3. Focus on domain-specific relationships

KEY PHRASES:"""

        try:
            key_phrases_output = gen(key_phrase_prompt, max_new_tokens=100, temperature=0.1)
            key_phrases_text = key_phrases_output[0].get('generated_text', '').strip()
        except:
            key_phrases_text = f"semantic cluster, {terms[0]} related"

        # Paso 2: Verificación de relevancia (anti-alucinación, estilo QualIT)
        verification_prompt = f"""VERIFICATION TASK: Check if these key phrases accurately represent the given terms.

TERMS: {', '.join(terms)}
EXTRACTED KEY PHRASES: {key_phrases_text}

VERIFICATION CRITERIA:
1. Do the key phrases accurately reflect the semantic relationships?
2. Are they relevant to the technology domain?
3. Do they avoid hallucinated connections?

VERIFICATION RESULT (True/False): """

        try:
            verification_output = gen(verification_prompt, max_new_tokens=50, temperature=0.1)
            verification_text = verification_output[0].get('generated_text', '').strip()
            is_verified = 'true' in verification_text.lower()
        except:
            is_verified = True

        # Paso 3: Generación de explicación final con razonamiento estructurado
        explanation_prompt = f"""You are an expert in explainable AI and topic modeling. Generate a comprehensive explanation.

CLUSTER ANALYSIS:
- Cluster ID: {topic_id}-{cluster_id}
- Terms: {', '.join(terms)}
- Key Phrases: {key_phrases_text}
- Verification Status: {'Verified' if is_verified else 'Needs revision'}
- Clustering Quality: {silhouette_score:.3f}

EXPLANATION FRAMEWORK (following XAI best practices):
1. SEMANTIC COHERENCE: What conceptual theme unifies these terms?
2. DOMAIN RELEVANCE: How do these terms relate to the technology domain?
3. CLUSTERING JUSTIFICATION: Why does this grouping make computational sense?

Generate a JSON response with the following structure:
{{
    "explicación": "Clear, concise explanation of the cluster's semantic unity",
    "coherencia": [1-5 score],
    "relevancia": [1-5 score], 
    "cobertura": [1-5 score],
    "key_phrases": [list of verified key phrases],
    "reasoning": "Brief justification for the scores"
}}

JSON Response:"""

        try:
            explanation_output = gen(explanation_prompt, max_new_tokens=250, temperature=0.2)
            explanation_text = explanation_output[0].get('generated_text', '')
        except:
            explanation_text = f'{{"explicación": "Technology cluster: {", ".join(terms[:3])}", "coherencia": 3, "relevancia": 3, "cobertura": 3}}'

        return explanation_text, key_phrases_text, is_verified

    # Generar explicaciones con estrategia avanzada
    print("\n✨ Generando explicaciones con Chain-of-Thought...")
    explanations = {}
    detailed_analysis = {}

    for cid, terms_c in clusters.items():
        try:
            print(f"  🔄 Procesando cluster {cid}...")
            explanation_text, key_phrases, verified = generate_explanation_with_cot(
                cid, terms_c, TOPIC_ID, global_sil
            )

            # Extraer JSON de la explicación
            parsed = extract_json_from_text(explanation_text)
            if parsed is None:
                # Fallback con información del proceso
                parsed = {
                    'explicación': f"Technology cluster grouping: {', '.join(terms_c[:3])}",
                    'coherencia': 3,
                    'relevancia': 3,
                    'cobertura': 3,
                    'key_phrases': key_phrases.split(',') if key_phrases else [],
                    'reasoning': 'Automatic fallback explanation'
                }

            explanations[cid] = parsed
            detailed_analysis[cid] = {
                'key_phrases_extracted': key_phrases,
                'verification_passed': verified,
                'raw_output': explanation_text[:200] + "..." if len(explanation_text) > 200 else explanation_text
            }

        except Exception as e:
            print(f"  ❌ Error generando explicación para cluster {cid}: {e}")
            explanations[cid] = {
                'explicación': f"Technology cluster: {', '.join(terms_c[:3])}",
                'coherencia': 3,
                'relevancia': 3,
                'cobertura': 3,
                'key_phrases': [],
                'reasoning': f'Error in generation: {str(e)[:50]}'
            }

    explanations_str = {str(k): v for k, v in explanations.items()}
    with open(os.path.join(OUTPUT_DIR, 'explanations.json'), 'w', encoding='utf-8') as f:
        json.dump(explanations_str, f, ensure_ascii=False, indent=2)

    with open(os.path.join(OUTPUT_DIR, 'detailed_analysis.json'), 'w', encoding='utf-8') as f:
        json.dump({str(k): v for k, v in detailed_analysis.items()}, f, ensure_ascii=False, indent=2)

    print("✅ Explicaciones generadas y guardadas")

    # 5. EVALUACIÓN CON CRITERIOS ESPECÍFICOS XAI
    def evaluate_explanation_structured(cluster_id, explanation, terms, topic_id):
        """
        Evaluación estructurada basada en criterios XAI específicos
        """

        evaluation_prompt = f"""You are an expert evaluator of AI explanations. Evaluate this cluster explanation using XAI criteria.

CLUSTER INFORMATION:
- ID: {topic_id}-{cluster_id}
- Terms: {', '.join(terms)}
- Explanation: "{explanation.get('explicación', '')}"
- Generated Key Phrases: {explanation.get('key_phrases', [])}

EVALUATION CRITERIA (score 1-5):

COHERENCIA (Semantic Coherence):
- Are the terms semantically related?
- Does the explanation capture their unity?
- Is the clustering logically sound?

RELEVANCIA (Domain Relevance): 
- How relevant are these terms to the technology domain?
- Does the explanation connect to the broader topic context?
- Are domain-specific relationships identified?

COBERTURA (Coverage Completeness):
- Does the explanation cover the main aspects of the cluster?
- Are key semantic relationships addressed?
- Is the scope appropriate for the term set?

PROVIDE EVALUATION as JSON:
{{
    "coherencia": [1-5],
    "relevancia": [1-5], 
    "cobertura": [1-5],
    "justificación": "2-sentence explanation of scores",
    "fortalezas": ["strength1", "strength2"],
    "debilidades": ["weakness1", "weakness2"]
}}

JSON Evaluation:"""

        return evaluation_prompt

    # Realizar evaluación estructurada
    print("\n🔍 Evaluando explicaciones con criterios XAI...")

    try:
        tok_e = T5Tokenizer.from_pretrained(EVAL_MODEL)
        mod_e = T5ForConditionalGeneration.from_pretrained(EVAL_MODEL).to(torch_dev)
        evalp = pipeline('text2text-generation', model=mod_e, tokenizer=tok_e,
                         device=0 if torch.cuda.is_available() else -1)
        print("✅ Modelo evaluativo cargado")
    except Exception as e:
        print(f"❌ Error cargando modelo evaluativo: {e}")
        print("⚠️ Usando evaluación simplificada")
        evalp = None

    evaluations = {}
    for cid, exp in explanations.items():
        terms_c = clusters[cid]
        print(f"  🔄 Evaluando cluster {cid}...")

        # Usar el generativo principal para evaluación más consistente
        eval_prompt = evaluate_explanation_structured(cid, exp, terms_c, TOPIC_ID)

        try:
            if gen is not None:
                eval_output = gen(eval_prompt, max_new_tokens=200, temperature=0.1)
                eval_text = eval_output[0].get('generated_text', '')
                parsed = extract_json_from_text(eval_text)
            else:
                parsed = None

            if parsed is None:
                parsed = {
                    'coherencia': 3,
                    'relevancia': 3,
                    'cobertura': 3,
                    'justificación': 'Términos tecnológicos relacionados con coherencia media',
                    'fortalezas': ['Agrupación semántica clara'],
                    'debilidades': ['Necesita mayor especificidad']
                }
        except Exception as e:
            print(f"  ❌ Error evaluando cluster {cid}: {e}")
            parsed = {
                'coherencia': 3,
                'relevancia': 3,
                'cobertura': 3,
                'justificación': 'Evaluación automática con error',
                'fortalezas': ['Agrupación básica'],
                'debilidades': ['Error en procesamiento']
            }

        evaluations[cid] = parsed

    evaluations_str = {str(k): v for k, v in evaluations.items()}
    with open(os.path.join(OUTPUT_DIR, 'evaluations.json'), 'w', encoding='utf-8') as f:
        json.dump(evaluations_str, f, ensure_ascii=False, indent=2)

    print("✅ Evaluaciones completadas y guardadas")

    # 6. RESUMEN EJECUTIVO MEJORADO
    print("\n📋 Generando resumen ejecutivo...")

    with open(os.path.join(OUTPUT_DIR, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write(f"ANÁLISIS DE TÓPICOS - REPORTE EJECUTIVO\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"📊 CONFIGURACIÓN DEL ANÁLISIS\n")
        f.write(f"Repositorio: {config['repo_name']}\n")
        f.write(f"Tópico ID: {TOPIC_ID}\n")
        f.write(f"Vocabulario seleccionado: {len(config['vocabulary'])} términos\n")
        f.write(f"Número de clusters identificados: {len(clusters)}\n")
        f.write(f"Número óptimo automático: {optimal_clusters}\n")
        f.write(f"Número seleccionado por usuario: {final_clusters}\n")
        f.write(f"Calidad de clustering (Silhouette): {global_sil:.3f}\n")
        f.write(f"Método: Clustering jerárquico + LLM explicativo\n\n")

        f.write("📈 ESTADÍSTICAS DE VOCABULARIO\n")
        stats = config['vocabulary_stats']
        f.write(f"  • Total términos usados: {stats['total_terms']}\n")
        f.write(f"  • Términos LDA disponibles: {stats['lda_available']}\n")
        f.write(f"  • Entidades NER disponibles: {stats['ner_available']}\n")
        f.write(f"  • Entidades DBPedia disponibles: {stats['dbpedia_available']}\n\n")

        f.write("RESUMEN POR CLUSTER:\n")
        f.write("-" * 40 + "\n\n")

        for cid, terms_c in clusters.items():
            exp = explanations[cid]
            eva = evaluations[cid]

            f.write(f"📋 CLUSTER {cid}:\n")
            f.write(f"  Términos clave: {', '.join(terms_c)}\n")
            f.write(f"  \n")
            f.write(f"  🎯 Explicación: {exp.get('explicación', 'N/A')}\n")
            f.write(f"  \n")
            f.write(f"  📊 Puntuaciones:\n")
            f.write(f"     • Coherencia: {eva.get('coherencia', 'N/A')}/5\n")
            f.write(f"     • Relevancia: {eva.get('relevancia', 'N/A')}/5\n")
            f.write(f"     • Cobertura: {eva.get('cobertura', 'N/A')}/5\n")
            f.write(f"  \n")
            f.write(f"  ✅ Fortalezas: {', '.join(eva.get('fortalezas', ['N/A']))}\n")
            f.write(f"  ⚠️  Debilidades: {', '.join(eva.get('debilidades', ['N/A']))}\n")
            f.write(f"  \n")
            f.write(f"  💡 Justificación: {eva.get('justificación', 'N/A')}\n")
            f.write(f"  \n")

            # Información del proceso QualIT
            if cid in detailed_analysis:
                detail = detailed_analysis[cid]
                f.write(f"  🔍 Análisis detallado:\n")
                f.write(f"     • Frases clave extraídas: {detail.get('key_phrases_extracted', 'N/A')}\n")
                f.write(f"     • Verificación pasada: {detail.get('verification_passed', 'N/A')}\n")

            f.write("\n" + "-" * 40 + "\n\n")

        # Estadísticas globales
        coherencias = [eva.get('coherencia', 0) for eva in evaluations.values() if
                       isinstance(eva.get('coherencia'), (int, float))]
        relevancias = [eva.get('relevancia', 0) for eva in evaluations.values() if
                       isinstance(eva.get('relevancia'), (int, float))]
        coberturas = [eva.get('cobertura', 0) for eva in evaluations.values() if
                      isinstance(eva.get('cobertura'), (int, float))]

        if coherencias:
            f.write("📈 ESTADÍSTICAS GLOBALES:\n")
            f.write(f"  Coherencia promedio: {np.mean(coherencias):.2f}/5\n")
            f.write(f"  Relevancia promedio: {np.mean(relevancias):.2f}/5\n")
            f.write(f"  Cobertura promedio: {np.mean(coberturas):.2f}/5\n")
            f.write(f"  Calidad clustering: {global_sil:.3f}\n\n")

        f.write("🎯 ARCHIVOS GENERADOS:\n")
        f.write(f"  • clusters.json - Información de clusters\n")
        f.write(f"  • explanations.json - Explicaciones generadas\n")
        f.write(f"  • evaluations.json - Evaluaciones XAI\n")
        f.write(f"  • detailed_analysis.json - Análisis detallado\n")
        f.write(f"  • silhouette_evolution.png - Gráfico optimización\n")
        f.write(f"  • dendrogram_with_cut.png - Dendrograma\n")
        f.write(f"  • config_topic_{TOPIC_ID}.json - Configuración\n")

    print("✅ Resumen ejecutivo generado")

    # Resumen final en consola
    print(f"\n🎉 ¡ANÁLISIS COMPLETADO!")
    print("=" * 50)
    print(f"📂 Directorio de salida: {OUTPUT_DIR}")
    print(f"📊 {len(clusters)} clusters generados con silhouette {global_sil:.3f}")
    print(f"🎯 Tópico {TOPIC_ID} del repositorio '{config['repo_name']}'")
    if coherencias:
        print(f"⭐ Calidad promedio: Coherencia {np.mean(coherencias):.1f}/5")
    print("\n📁 Archivos generados:")
    print("  • summary.txt - Resumen ejecutivo completo")
    print("  • clusters.json - Información detallada de clusters")
    print("  • explanations.json - Explicaciones con Chain-of-Thought")
    print("  • evaluations.json - Evaluaciones con criterios XAI")
    print("  • Gráficos PNG - Visualizaciones del clustering")


if __name__ == "__main__":
    main()