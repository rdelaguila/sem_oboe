import pandas as pd
import re
import time
import socket
from functools import lru_cache
from urllib.error import URLError, HTTPError
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from spotlight import annotate
from SPARQLWrapper import SPARQLWrapper, JSON, CSV
from owlready2 import World
import spacy
import nltk

class TextAnalyzer:
    def __init__(self, nlp):
        self.nlp = nlp

    @staticmethod
    def remove_special_lines(text: str) -> str:
        text = re.sub(r"^upright=.*[\r\n]", "", text, flags=re.MULTILINE)
        text = re.sub(r"Category:.*[\r\n]", "", text, flags=re.MULTILINE)
        text = re.sub(r"\[\d+\]", "", text)
        text = re.sub(r"thumb|\d+px|\|", "", text)
        return text

    @staticmethod
    def strip_formatting(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[\.!?,;\-_'/|()=<>+*`]", "", text)
        text = re.sub(r"https?://\S+", "", text)
        return text

    def __call__(self, doc: str):
        tokens = self.nlp(doc)
        return [
            token.lemma_.lower()
            for token in tokens
            if not (token.is_stop or token.is_punct)
        ]

    def is_present(self, word: str, text: str) -> bool:
        normalized = " ".join(
            token.lemma_.lower() for token in self.nlp(text)
            if not (token.is_stop or token.is_punct)
        )
        return word in normalized

class SemanticAnalyzer(TextAnalyzer):
    def __init__(
        self,
        nlp,
        endpoint: str = "http://localhost:2222/rest/annotate",
        soporte: int = 1000,
        confianza: float = 0.5,
        umbral: float = 0.1
    ):
        super().__init__(nlp)
        self.endpoint = endpoint
        self.soporte = soporte
        self.confianza = confianza
        self.umbral = umbral

    def __call__(self, doc: str) -> dict:
        try:
            annotations = annotate(
                self.endpoint,
                doc,
                confidence=self.confianza,
                support=self.soporte
            )
            return {
                ann['surfaceForm']: {
                    'URI': ann['URI'],
                    'types': ann.get('types', [])
                }
                for ann in annotations
            }
        except Exception as ex:
            print(f"Spotlight error: {ex}")
            return {}
class OntoManager:
    def __init__(
        self,
        nlp,
        path_dbo: str = "file://./data/ontologias/dbpedia_2016-10.owl/",
        path_sumo: str = "file://./data/ontologias/SUMO.owl",
        sparql_endpoint: str = "https://dbpedia.org/sparql",
        timeout_ms: int = 10000
    ):
        self.nlp = nlp
        self.sa = SemanticAnalyzer(nlp)
        self.prefijos = """
            PREFIX rdfs:<http://www.w3.org/2000/01/rdf-schema#>
            PREFIX rdf:<http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX dbr:<http://dbpedia.org/resource/>
            PREFIX dbo:<http://dbpedia.org/ontology/>
            PREFIX owl:<http://www.w3.org/2002/07/owl#>
        """
        self.dict_onto, self.dict_graph = self._load_ontologies(path_dbo, path_sumo)
        self.sparql_endpoint = sparql_endpoint
        self.timeout_ms = timeout_ms

    def _load_ontologies(self, path_dbo, path_sumo):
        w1, w2 = World(), World()
        onto_sumo = w1.get_ontology(path_sumo).load()
        onto_dbo  = w2.get_ontology(path_dbo).load()
        return (
            {'SUMO': onto_sumo, 'dbo': onto_dbo},
            {'SUMO': w1.as_rdflib_graph(), 'dbo': w2.as_rdflib_graph()}
        )

    def _isDbo(self, term: str) -> bool:
        if self.dict_onto['dbo'].search(label=term, _case_sensitive=False):
            return True
        if self.dict_onto['SUMO'].search(label=term, _case_sensitive=False):
            return False
        return None

    def _getBaseConcept(self, term: str, isDbo: bool) -> str:
        onto = self.dict_onto['dbo'] if isDbo else self.dict_onto['SUMO']
        res = onto.search(label=term, _case_sensitive=False)
        return str(res[0]).replace('.', ':') if res else None

    @lru_cache(maxsize=5000)
    def _getDBPediaTypes(self, uri_q: str, retries: int = 3, backoff_sec: float = 1.0) -> list:
        """
        Realiza consulta SPARQL con timeout, reintentos y caché.
        Espera recibir la URI entre '<' y '>'.
        """
        # Construir consulta
        q = f"{self.prefijos} SELECT DISTINCT ?o WHERE {{ {uri_q} rdf:type ?o }}"
        # Intentar carga desde grafo interno primero
        results = list(self.dict_graph['dbo'].query(q))
        if results:
            return [str(r[0]) for r in results]

        # Si no hay resultados locales, consultar endpoint externo
        sparql = SPARQLWrapper(self.sparql_endpoint)
        sparql.setTimeout(self.timeout_ms)
        sparql.setReturnFormat(CSV)
        sparql.setQuery(q)

        for attempt in range(1, retries + 1):
            try:
                csv_res = sparql.query().convert()
                # parse CSV: eliminar comillas y línea de cabecera
                lines = str(csv_res).replace('"', '').split('\n')[1:]
                return [line for line in lines if line]
            except (socket.timeout, URLError, HTTPError) as e:
                if attempt < retries:
                    time.sleep(backoff_sec * 2**(attempt-1))
                else:
                    print(f"Advertencia SPARQL para {uri_q}: {e}")
                    return []

    def getSemanticsOfTerm(self, term: str) -> dict:
        is_dbo = self._isDbo(term)
        if is_dbo is None:
            return None
        concept   = self._getBaseConcept(term, is_dbo)
        resources = self.sa(term) if is_dbo else {}
        types     = self._getDBPediaTypes(f"<{concept.replace('dbr:', 'http://dbpedia.org/resource/')}>")
        return {
            'concepto': concept,
            'tipos': types,
            'resources': resources
        }

    def normalize(self, input_data):
        if isinstance(input_data, (list, set, tuple)):
            return [self.normalize(x) for x in input_data]
        text = str(input_data)
        doc  = self.nlp(text)
        tokens = [
            token.lemma_.lower()
            for token in doc
            if not token.is_punct and not token.is_space
        ]
        return " ".join(tokens)

# Función auxiliar para consultas en paralelo
def fetch_all_types(onto_manager: OntoManager, uris: set[str], max_workers: int = 5) -> dict:
    lookup: dict[str, list] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_uri = {executor.submit(onto_manager._getDBPediaTypes, f"<{uri}>"): uri for uri in uris}
        for future in as_completed(future_to_uri):
            uri = future_to_uri[future]
            try:
                lookup[uri] = future.result()
            except Exception as e:
                print(f"Error obteniendo tipos para {uri}: {e}")
                lookup[uri] = []
    return lookup

# Simplificación y limpieza de entidades

def simplificar_entidades(entidad):
    return {entidad.get('surfaceForm'): {'URI': entidad.get('URI'), 'tipos': entidad.get('types', [])}}

def limpiar_entidades(entidades_series):
    entidades_simplificadas = {}
    for entidad in entidades_series:
        entidades_simplificadas.update(simplificar_entidades(entidad))
    return entidades_simplificadas

# Completar tipos faltantes usando consulta en lote

def completar_tipos_faltantes(entidades_dict, onto_manager):
    # 1) Identificar URIs sin tipos
    uris_faltantes = {props['URI'] for props in entidades_dict.values() if not props['tipos']}
    # 2) Obtener en lote todos los tipos faltantes
    tipos_lookup = fetch_all_types(onto_manager, uris_faltantes)
    # 3) Asignar resultados de vuelta
    for termino, props in entidades_dict.items():
        if not props['tipos']:
            props['tipos'] = tipos_lookup.get(props['URI'], [])
    return entidades_dict
