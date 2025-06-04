import pandas as pd
import joblib
corpus1 = joblib.load('./bbc_objects/bbc_processed_final_semantic')
cont = 0
def simplificar_entidades(entidad):
    diccionario = dict()
    #diccionario['URI'] = entidad.get('URI')
    #diccionario['surfaceForm'] = entidad.get('surfaceForm'),  
    #diccionario['tipos']: entidad.get('types') if entidad.keys() else None 
    diccionario[entidad.get('surfaceForm')] = dict({'URI':entidad.get('URI'), 'tipos':entidad.get('types') if entidad.keys() else None })
    return diccionario

def devolver_entidades_limpias_df (row):
    entidades_dbpedia = pd.Series(row['entidades_dbpedia'])
   # global cont
   # cont = cont + 1 
   # print(cont)
    dicts = entidades_dbpedia.apply(simplificar_entidades)
    print(len(dicts))
    import collections
    entidades_simplificadas = {}
    for d in dicts:
        for k, v in d.items():  # d.items() in Python 3+
            entidades_simplificadas[k] = v
    return entidades_simplificadas

corpus1['entidades_dbpedia_simplificadas'] = corpus1.apply(devolver_entidades_limpias_df,axis=1)

#cargamos dbpedia y sumo
from spotlight import *
from SPARQLWrapper import SPARQLWrapper, JSON, CSV

class TextAnalyzer(object):
    def __init__(self,nlp):
        self.nlp = nlp 
        
    # allow the class instance to be called just like
    # just like a function and applies the preprocessing and
    # tokenize the document
    @staticmethod      
    def remove_special_lines(texto):
        texto = re.sub("^upright=.*[\r|\n]", '', texto)
        texto = re.sub("^upright = .*[\r|\n]", '', texto)
        texto = re.sub("Category:.*[\r|\n]",'',texto)
        texto = re.sub("Cat\D*:.*[\r|\n]",'',texto)
        texto = re.sub("[[][\d]+[]]",'',texto)
        texto = re.sub("thumb",'',texto)
        texto = re.sub("[|]",'',texto)
        texto = re.sub("\d+px",'',texto)
        return (texto)
    @staticmethod
    def strip_formatting(string):
        string = string.lower()
        string = re.sub(r"([.!?,;-_'/|()]=-<>+*`)", r"", string)
        string = re.sub(r'https?:\/\/.*?[\s]', '', string) 
        return string

    def get_nlp(self):
        return self.nlp
    
    def __call__(self, doc):
        tokens = nlp(doc)
        lemmatized_tokens = [(token.lemma_.lower()) for token in tokens
                                                   if not (token.is_stop or token.is_punct)]
            
        return(lemmatized_tokens)
    
    def is_present (self,word,text):
        lemmatized_tokens =  lambda text: " ".join(token.lemma_.lower() for token in nlp(text) if not (token.is_stop or token.is_punct))
        normalizado = lemmatized_tokens(text)    
        return (word in (normalizado))

class SemanticAnalyzer(TextAnalyzer):
    def __init__(self,nlp,endpoint="http://172.17.0.1:2222/rest/annotate",soporte=1000,confianza=0.5,umbral=0.1):
        super().__init__(nlp)
        self.endpoint = endpoint
        self.soporte=soporte
        self.confianza = confianza
        self.alfa = umbral
    
    def __call__(self, doc):
        try:
            annotations = spotlight.annotate(self.endpoint,
                                     doc,
                                      confidence=self.confianza, support=self.soporte, spotter='Default')
            diccionario =  dict()
            for annotation in annotations:
                lista = list(annotation.items())
                print(lista)
                URI = lista[0]
                key = lista[3]
                score = lista[5]
               # if (score[1]>self.alfa):
                diccionario[key[1]]=URI[1]
        
            return(diccionario)
        except Exception as ex:
            print(ex)

import spacy
import time 
from owlready2 import *
import json 
from spotlight import *

class OntoManager(object):
    def __init__(self,nlp,dict_onto,dict_graph):
        
        self.dict_onto = dict_onto
        self.dict_graph = dict_graph
        
        self.prefijos = """  PREFIX rdfs:<http://www.w3.org/2000/01/rdf-schema#>
                                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                                PREFIX dbr:    <http://dbpedia.org/resource/>
                                PREFIX dbo:    <http://dbpedia.org/ontology/>
                                PREFIX dct:    <http://purl.org/dc/terms/>
                                PREFIX owl:    <http://www.w3.org/2002/07/owl#>
                                PREFIX rdf:    <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                                PREFIX rdfs:   <http://www.w3.org/2000/01/rdf-schema#>
                                PREFIX schema: <http://schema.org/>
                                PREFIX skos:   <http://www.w3.org/2004/02/skos/core#>
                                PREFIX xsd:    <http://www.w3.org/2001/XMLSchema#>
                                PREFIX SUMO: <http://www.adampease.org/OP/SUMO.owl#>
                            """
        self.sa = SemanticAnalyzer(nlp)
        
    def getSemanticsOfTerm(self,term):   
        isDbo = self._isDbo(term)
        if (isDbo is None):
            return None
        
        concept = self._getBaseConcept(term,isDbo)
        if (isDbo):
            resources = self.sa(term)
        else:
            resources = {}
        superclasses = self._getHierarchy(str(concept),isDbo,False)
        subclasses = self._getHierarchy(str(concept),isDbo,True)
        relationships = self._getRelationships(str(concept),isDbo)
        types = self._getDBPediaTypes(str(concept))
        
        termino = dict({'concepto':concept, 'tipos':types,'resources':resources,'padres':superclasses,'hijos':subclasses,'relaciones':relationships})
        
        return termino

    from SPARQLWrapper import SPARQLWrapper, JSON, CSV


    def ejecutar_consulta_dbpedia(self,query,tipo=JSON):
        sparql = SPARQLWrapper("http://dbpedia.org/sparql")
        sparql.setReturnFormat(tipo)

        sparql.setQuery(self.prefijos+query)  # the previous query as a literal string
        print(query)
        return sparql.query().convert()

    #def getTypesOfURI (self, uri):
        
    def _isDbo (self,term):
        res = self.dict_onto.get('dbo').search(label=term,_case_sensitive=False)
        if (len(res)>0):
            return True
        res = self.dict_onto.get('SUMO').search(label=term,_case_sensitive=False)
        if (len(res)>0):
            return False
        else:
            return None

    def _getBaseConcept(self,term,isDbo):
        if (isDbo):
            res = self.dict_onto.get('dbo').search(label=term,_case_sensitive=False)
            if (len(res)>0):
                return str(res[0]).replace('.',':')
        
        res = self.dict_onto.get('SUMO').search(label=term,_case_sensitive=False)
        if (len(res)>0):
            return str(res[0]).replace('.',':')
        else:
            return None
    
    def _getDBPediaTypes(self, term):
        consulta = self.prefijos + """
       select ?o where {"""+term.replace('dbo','dbr')+""" rdf:type ?o}

        """
        print(consulta)
        lista = list(self.dict_graph.get('dbo').query(consulta))
        if len(lista)==0:
            res = self.ejecutar_consulta_dbpedia(consulta, CSV)
            lista = str(res).replace("""\"""",'').split('\\n')[1:]
        return lista
        
    def _getHierarchy(self,concept,isDbo,isSuperclass):
        
        consulta = self.prefijos + """SELECT ?x
            WHERE {
                ?x a owl:Class .
                ?x rdfs:subClassOf """+concept+"""
                }"""
         
        if isSuperclass == False:
                 consulta = self.prefijos + """SELECT ?x
            WHERE {
                ?x a owl:Class .
                """+concept+""" rdfs:subClassOf ?x 
                }"""
        if (isDbo==False):
            return (list(self.dict_graph.get('SUMO').query(consulta)))  
        else:
            return (list(self.dict_graph.get('dbo').query(consulta)))
    
    def _getRelationships(self,concept,isDbo):
        consulta = self.prefijos + """select distinct * where {
                  """+concept+""" ?property ?value .
                  filter ( ?property not in ( rdf:type ) )
                   filter ( ?property not in ( rdfs:label ) )
                optional {?property rdfs:comment ?comment}
                  optional {?property rdfs:label ?label}
                  optional {?property rdfs:range ?range} 
                  optional {?property rdfs:domain ?domain} 
                }
        """
        
        if (isDbo==False):
            resultado =  (list(self.dict_graph.get('SUMO').query(consulta)))  
        else:
            resultado = (list(self.dict_graph.get('dbo').query(consulta)))
        
        res = pd.DataFrame(data = resultado, columns = ['term','property','comment','label','range','domain'])
        res.set_index(res.term)
        return (res.to_json())
        
from owlready2 import *
import pandas as pd
nlp = spacy.load("en_core_web_md")
myworld1 = World()
sumo =myworld1.get_ontology("file:///home/raul/doctorado/ontologias/SUMO.owl").load()
graphsumo = myworld1.as_rdflib_graph()
myworld2 = World()
dbpedia = myworld2.get_ontology("file:///home/raul/doctorado/ontologias/dbpedia_3.9.owl/").load()
graphdbo = myworld2.as_rdflib_graph()
#dbpedia.base_iri = "http://dbpedia.org/ontology/"
dbpedia.name='dbo'
dict_onto = dict([('dbo',dbpedia),('SUMO',sumo)])
dict_graph = dict([('dbo',graphdbo),('SUMO',graphsumo)])
alfred = OntoManager(nlp,dict_onto,dict_graph)

import pandas as pd
#corpus1 = joblib.load('./bbc_objects/bbc_processed_final_semantic')
cont = 0

def devolver_nuevos_tipos (row):
    
    cluster_dict = defaultdict(list)
    for termino, propiedades in row['entidades_dbpedia_simplificadas'].items():
        
        new_props = propiedades
        if new_props['tipos']==[] or len(new_props['tipos'])==0:
            dbr = '<'+new_props['URI']+'>'#.replace("http://dbpedia.org/resource/","dbr:")
            #dbr = dbr[0:len(dbr)-1] if dbr.endswith('.') else dbr
            #dbr = dbr.replace("'","""\\'""") if dbr.find("'")>-1 else dbr
            new_props['tipos']=alfred._getDBPediaTypes(dbr)
        cluster_dict[termino].append(new_props)
        
    return cluster_dict

corpus1['entidades_dbpedia_simplificadas'] = corpus1.apply(devolver_nuevos_tipos,axis=1)
import time
start_time = time.time()

joblib.dump(corpus1,'./bbc_objects/bbc_processed_final_semantic_2')

print('duracion {0}'.format(time.time())) 