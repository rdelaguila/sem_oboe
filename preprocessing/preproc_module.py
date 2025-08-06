import re
import time
import socket
from functools import lru_cache
from urllib.error import URLError, HTTPError
from threading import Lock
from collections import defaultdict
from spotlight import annotate
from SPARQLWrapper import SPARQLWrapper, JSON, CSV
from owlready2 import World
import pandas as pd


class TextAnalyzer:
    def __init__(self, nlp):
        self.nlp = nlp

    @staticmethod
    def remove_special_lines(text: str) -> str:
        text = re.sub(r"^upright=.*[\r\n]", "", text, flags=re.MULTILINE)
        text = re.sub(r"^upright = .*?[\r\n]", "", text, flags=re.MULTILINE)
        text = re.sub(r"Category:.*?[\r\n]", "", text, flags=re.MULTILINE)
        text = re.sub(r"Cat\D*:.*?[\r\n]", "", text, flags=re.MULTILINE)
        text = re.sub(r"\[\d+\]", "", text)
        text = re.sub(r"thumb", "", text)
        text = re.sub(r"[|]", "", text)
        text = re.sub(r"\d+px", "", text)
        return text

    @staticmethod
    def strip_formatting(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[\.\!\?,;\-_'/|()=<>+*`]", "", text)
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

    # NUEVA FUNCIÓN: Extracción de entidades con spaCy NER
    def extract_spacy_entities(self, text: str) -> set:
        """
        Extrae entidades nombradas usando spaCy NER,
        filtrando tipos no deseados como en el código original
        """
        doc = self.nlp(text)
        entidades = set()

        if doc.ents:
            for ent in doc.ents:
                # Filtrar tipos como en el código original
                if ent.label_ not in ['ORDINAL', 'CARDINAL', 'TIME']:
                    entidades.add(ent.text.lower())

        return entidades


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
            result = {}
            for ann in annotations:
                surface = ann.get('surfaceForm')
                uri = ann.get('URI')
                types_str = ann.get('types', '') or ''
                types_list = types_str.split(',') if types_str else []
                # normalize type labels
                types_norm = [self.strip_formatting(t.replace(':', ' ')) for t in types_list]
                result[surface] = {
                    'URI': uri,
                    'types': ",".join(types_list),
                    'types_normalizados': ",".join(types_norm)
                }
            return result
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
        self.prefijos = (
            "PREFIX rdfs:<http://www.w3.org/2000/01/rdf-schema#>"
            "PREFIX rdf:<http://www.w3.org/1999/02/22-rdf-syntax-ns#>"
            "PREFIX dbr:<http://dbpedia.org/resource/>"
            "PREFIX dbo:<http://dbpedia.org/ontology/>"
            "PREFIX owl:<http://www.w3.org/2002/07/owl#>"
        )
        self.dict_onto, self.dict_graph = self._load_ontologies(path_dbo, path_sumo)
        self.sparql_endpoint = sparql_endpoint
        self.timeout_ms = timeout_ms
        self._sparql_lock = Lock()

    def _load_ontologies(self, path_dbo, path_sumo):
        w1, w2 = World(), World()
        onto_sumo = w1.get_ontology(path_sumo).load()
        onto_dbo = w2.get_ontology(path_dbo).load()
        return (
            {'SUMO': onto_sumo, 'dbo': onto_dbo},
            {'SUMO': w1.as_rdflib_graph(), 'dbo': w2.as_rdflib_graph()}
        )

    def ejecutar_consulta_dbpedia(self, query: str, tipo=JSON):
        sparql = SPARQLWrapper(self.sparql_endpoint)
        sparql.setReturnFormat(tipo)
        sparql.setQuery(self.prefijos + query)
        return sparql.query().convert()

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
    def _getDBPediaTypes(self, uri: str) -> list:
        consulta = (
                self.prefijos +
                f"SELECT DISTINCT ?o WHERE {{ {uri} rdf:type ?o }}"
        )
        local = list(self.dict_graph['dbo'].query(consulta))
        if local:
            return [str(r[0]) for r in local]
        sparql = SPARQLWrapper(self.sparql_endpoint)
        sparql.setTimeout(self.timeout_ms)
        sparql.setReturnFormat(CSV)
        sparql.setQuery(consulta)
        try:
            csv_res = sparql.query().convert()
            lines = str(csv_res).replace('"', '').split('\n')[1:]
            return [l for l in lines if l]
        except Exception as e:
            print(f"SPARQL warning for {uri}: {e}")
            return []

    def _getHierarchy(self, concept: str, isDbo: bool, isSuperclass: bool) -> list:
        if isSuperclass:
            q = (
                    self.prefijos +
                    f"SELECT ?x WHERE {{ ?x a owl:Class; rdfs:subClassOf {concept} }}"
            )
        else:
            q = (
                    self.prefijos +
                    f"SELECT ?x WHERE {{ {concept} rdfs:subClassOf ?x }}"
            )
        graph = self.dict_graph['dbo'] if isDbo else self.dict_graph['SUMO']
        return [str(r[0]) for r in graph.query(q)]

    def _getRelationships(self, concept: str, isDbo: bool) -> list:
        consulta = (
                self.prefijos +
                f"SELECT DISTINCT * WHERE {{ {concept} ?property ?value ."
                "FILTER(?property NOT IN (rdf:type, rdfs:label))"
                "OPTIONAL { ?property rdfs:comment ?comment }"
                "OPTIONAL { ?property rdfs:label ?label }"
                "OPTIONAL { ?property rdfs:range ?range }"
                "OPTIONAL { ?property rdfs:domain ?domain }"
                "}"
        )
        graph = self.dict_graph['dbo'] if isDbo else self.dict_graph['SUMO']
        rows = list(graph.query(consulta))
        df = pd.DataFrame(rows, columns=['term', 'property', 'comment', 'label', 'range', 'domain'])
        return df.to_dict(orient='records')

    def getSemanticsOfTerm(self, term: str) -> dict:
        is_dbo = self._isDbo(term)
        if is_dbo is None:
            return None
        concept = self._getBaseConcept(term, is_dbo)
        resources = self.sa(term) if is_dbo else {}
        types = self._getDBPediaTypes(f"<{concept.replace('dbr:', 'http://dbpedia.org/resource/')}>")
        # normalize types
        types_norm = [TextAnalyzer.strip_formatting(t.replace(':', ' ')) for t in types]
        superclasses = self._getHierarchy(concept, is_dbo, True)
        subclasses = self._getHierarchy(concept, is_dbo, False)
        relationships = self._getRelationships(concept, is_dbo)
        return {
            'concepto': concept,
            'tipos': types,
            'types_normalizados': types_norm,
            'resources': resources,
            'padres': superclasses,
            'hijos': subclasses,
            'relaciones': relationships
        }

    def normalize(self, input_data):
        if isinstance(input_data, (list, set, tuple)):
            return [self.normalize(x) for x in input_data]
        text = str(input_data)
        doc = self.nlp(text)
        tokens = [
            token.lemma_.lower()
            for token in doc
            if not token.is_punct and not token.is_space
        ]
        return " ".join(tokens)


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


def fetch_all_types(onto_manager, uris_set):
    """Función auxiliar para obtener tipos en lote"""
    tipos_lookup = {}
    for uri in uris_set:
        try:
            tipos_lookup[uri] = onto_manager._getDBPediaTypes(f"<{uri}>")
        except Exception as e:
            print(f"Error obteniendo tipos para {uri}: {e}")
            tipos_lookup[uri] = []
    return tipos_lookup

