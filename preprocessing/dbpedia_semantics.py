
import pandas as pd
import re
from collections import defaultdict
from spotlight import annotate
from SPARQLWrapper import SPARQLWrapper, JSON, CSV
from owlready2 import World
import spacy

class SemanticAnalyzer:
    def __init__(self, nlp, endpoint="http://localhost:2222/rest/annotate", soporte=1000, confianza=0.5, umbral=0.1):
        self.nlp = nlp
        self.endpoint = endpoint
        self.soporte = soporte
        self.confianza = confianza
        self.umbral = umbral

    def __call__(self, doc):
        try:
            annotations = annotate(self.endpoint, doc, confidence=self.confianza, support=self.soporte)
            return {a['surfaceForm']: {'URI': a['URI'], 'types': a.get('types', [])} for a in annotations}
        except Exception as ex:
            print(f"Annotation error: {ex}")
            return {}

class OntoManager:
    def __init__(self, nlp, path_dbo, path_sumo):
        self.sa = SemanticAnalyzer(nlp)
        self.prefijos = """
            PREFIX rdfs:<http://www.w3.org/2000/01/rdf-schema#>
            PREFIX rdf:<http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX dbr:<http://dbpedia.org/resource/>
            PREFIX dbo:<http://dbpedia.org/ontology/>
            PREFIX owl:<http://www.w3.org/2002/07/owl#>
        """
        self.dict_onto, self.dict_graph = self._load_ontologies(path_dbo, path_sumo)

    def _load_ontologies(self, path_dbo, path_sumo):
        w1, w2 = World(), World()
        onto_sumo = w1.get_ontology(path_sumo).load()
        onto_dbo = w2.get_ontology(path_dbo).load()
        graph_sumo = w1.as_rdflib_graph()
        graph_dbo = w2.as_rdflib_graph()
        return {'SUMO': onto_sumo, 'dbo': onto_dbo}, {'SUMO': graph_sumo, 'dbo': graph_dbo}

    def _isDbo(self, term):
        if self.dict_onto['dbo'].search(label=term, _case_sensitive=False):
            return True
        if self.dict_onto['SUMO'].search(label=term, _case_sensitive=False):
            return False
        return None

    def _getBaseConcept(self, term, isDbo):
        onto = self.dict_onto['dbo'] if isDbo else self.dict_onto['SUMO']
        res = onto.search(label=term, _case_sensitive=False)
        return str(res[0]).replace('.', ':') if res else None

    def _getDBPediaTypes(self, term):
        term = term.replace('dbo:', 'dbr:')
        consulta = f"{self.prefijos} SELECT ?o WHERE {{{term} rdf:type ?o}}"
        results = list(self.dict_graph['dbo'].query(consulta))
        if not results:
            sparql = SPARQLWrapper("http://dbpedia.org/sparql")
            sparql.setQuery(consulta)
            sparql.setReturnFormat(CSV)
            result = sparql.query().convert()
            return str(result).replace('"', '').split('\n')[1:]
        return results

    def getSemanticsOfTerm(self, term):
        isDbo = self._isDbo(term)
        if isDbo is None:
            return None
        concept = self._getBaseConcept(term, isDbo)
        if not concept:
            return None
        resources = self.sa(term) if isDbo else {}
        types = self._getDBPediaTypes(concept)
        return {'concepto': concept, 'tipos': types, 'resources': resources}

def simplificar_entidades(entidad):
    return {entidad.get('surfaceForm'): {'URI': entidad.get('URI'), 'tipos': entidad.get('types', [])}}

def limpiar_entidades(entidades_series):
    entidades_simplificadas = {}
    for entidad in entidades_series:
        entidades_simplificadas.update(simplificar_entidades(entidad))
    return entidades_simplificadas

def completar_tipos_faltantes(entidades_dict, onto_manager):
    cluster_dict = defaultdict(list)
    for termino, props in entidades_dict.items():
        if not props['tipos']:
            uri = f"<{props['URI']}>"
            props['tipos'] = onto_manager._getDBPediaTypes(uri)
        cluster_dict[termino].append(props)
    return cluster_dict
