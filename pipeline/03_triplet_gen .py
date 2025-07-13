#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import stanza
import stanza

# In[2]:


from stanza.server import CoreNLPClient

client = CoreNLPClient(
    annotators=['openie'],
    endpoint='http://0.0.0.0:9000',
    start_server=False,
    be_quiet=True)
print(client)

# In[3]:


import joblib
import dask.dataframe as dd

# In[4]:


df = joblib.load('../bbc_objects/bbc_processed_final_semantic_2')
ddf = dd.from_pandas(df, npartitions=10)


def devolver_tripletas(sentence, frase, pos, tripletgenerator):
    tripletas = []
    tripletas = [tripletgenerator.encapsular((triple.subject, triple.relation, triple.object), True) for triple in
                 sentence.openieTriple]  # encapsular triplet generator
    if len(tripletas) == 0:
        tripleta = tripletgenerator.triplet_extraction(frase, True)
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
        # print('devuekvi {}'.format(tripleta))
        if tripleta is not None:
            tripleta = tripletgenerator.encapsular(tripleta, True)
            if type(tripleta) == list:
                tripletas.extend(tripleta)
            else:
                tripletas.append(tripleta)
    if tripletgenerator._detect_nn_of_nn(pos)[0] != -1:
        tripleta = tripletgenerator.generar_tripleta_nn_of_nn(pos)
        if tripleta is not None:
            # print (tripleta[0])
            # print(tripleta[1])
            # print (tripleta[2])
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
        # print ('0'+str(tripleta))
        tripleta = tripletgenerator.encapsular(tripleta, True)
        if type(tripleta) == list:
            tripletas.extend(tripleta)
        else:
            tripletas.append(tripleta)
    return tripletas
    # for triple in sentence.openieTriple:
    #    print(triple.subject,triple.relation,triple.object)


def devolver_pos(sentence):
    diccionario = dict()
    for posicion in (sentence.ListFields()[0][1]):
        diccionario[posicion.word] = posicion.pos
    return diccionario

    # for triple in sentence.openieTriple:
    #    print(triple.subject,triple.relation,triple.object)


def transformar(df):
    return df.apply(anotar, axis=1)


def anota_texto(text):
    return client.annotate(text)


# def anotacion_vectorizada(x):
#    return np.vectorize(anota_texto)(x)
import traceback

import importlib
import semantic_oboe

importlib.reload(semantic_oboe)
topics = joblib.load('/Users/Raul/doctorado/semantic_oboe/semantic_oboe/bbc_objects/new_bbc_topics_7_sinprob')

from semantic_oboe.triplet_manager_lib import *
from time import sleep


def anotar(row):
    tripletgenerator = TripletGenerator()
    resultado = []
    tripletmanager = TripletManager()
    row_topic = row['new_target']
    dbpedia = list(row['entidades_dbpedia_simplificadas'].keys())

    ner = row['entidades']
    topic = topics.get(row_topic)

    for frase in row['text'].split('.'):
        try:

            anotacion = client.annotate(frase)
            sleep(.30)

            for sentence in anotacion.sentence:
                d = devolver_pos(sentence)
                t = devolver_tripletas(sentence, frase, d, tripletgenerator)
                # tripletas.append(t)
                # diccionario.append(d)
                # tupla_resultado = (d,t)
                # resultado.extend(tripletmanager.es_candidata([tripleta for tripleta in t if tripletmanager.es_candidata(tripleta,d,ner,topic,dbpedia)]))
                resultado.extend(t)
        except:
            continue

    return resultado


#   except:
#           print(type(sentence))
##           print(t)
#          print('******')
#          return


ddf['tripletas'] = ddf.map_partitions(transformar, meta=('object')).compute()

# In[ ]:

print('proceso finalizado, se procede a guardar')
df = ddf.compute()
joblib.dump(ddf.compute(), '../bbc_objects/bbc_semantic_tripletas_simplificado')
print('guardado')

print('Transformando a dataframe de tripletas')

df = joblib.load('../bbc_objects/bbc_semantic_tripletas_simplificado')

df.drop(df[df['tripletas'].map(len) == 0].index, inplace=True)

triplet_df_def = df.explode('tripletas')
triplet_df_def = triplet_df_def.reset_index()
triplet_df_kge = pd.DataFrame(triplet_df_def['tripletas'].tolist(), index=triplet_df_def.index)

triplet_df_kge['new_topic'] = triplet_df_def.new_target
triplet_df_kge['old_index'] = triplet_df_def.index
triplet_df_kge.columns = ['subject', 'relation', 'object', 'new_topic', 'old_index']
triplet_df_kge.to_csv('../bbc_objects/dataset_triplet_bbc_new_simplificado.csv')
print('Finalizado transformacion y guardado en {}'.format('../bbc_objects/dataset_triplet_bbc_new_simplificado.csv'))