
import time
from stanza.server import CoreNLPClient
from semantic_oboe.triplet_manager_lib import TripletGenerator
from collections import defaultdict

# Inicialización del cliente de CoreNLP
client = CoreNLPClient(
    annotators=['openie'],
    endpoint='http://0.0.0.0:9000',
    start_server=False,
    be_quiet=True
)

def devolver_pos(sentence):
    return {token.word: token.pos for token in sentence.ListFields()[0][1]}

def devolver_tripletas(sentence, frase, pos, tripletgenerator):
    tripletas = [tripletgenerator.encapsular((triple.subject, triple.relation, triple.object), True)
                 for triple in sentence.openieTriple]

    if not tripletas:
        tripleta = tripletgenerator.triplet_extraction(frase, True)
        if tripleta:
            tripletas.append(tripletgenerator.encapsular(tripleta, True))

    for detect, generar in [
        (tripletgenerator._detect_adj_nou, tripletgenerator.generar_tripleta_adj_noun),
        (tripletgenerator._detect_nn_nnp, tripletgenerator.generar_tripleta_nn_nnp),
        (tripletgenerator._detect_nn_of_nn, tripletgenerator.generar_tripleta_nn_of_nn),
        (tripletgenerator._detect_nn_place_nn, tripletgenerator.generar_tripleta_nn_place_nn)
    ]:
        if detect(pos)[0] != -1:
            tripleta = generar(pos)
            if tripleta:
                encapsulada = tripletgenerator.encapsular(tripleta, True)
                tripletas.extend(encapsulada if isinstance(encapsulada, list) else [encapsulada])

    adj_trip = tripletgenerator.generar_tripleta_adjectives_nn_adjs(pos)
    if adj_trip:
        encapsulada = tripletgenerator.encapsular(adj_trip, True)
        tripletas.extend(encapsulada if isinstance(encapsulada, list) else [encapsulada])

    return tripletas

def anotar_fila(row, tripletmanager, topics):
    resultado = []
    tripletgenerator = TripletGenerator()
    row_topic = row['new_target']
    dbpedia = list(row['entidades_dbpedia_simplificadas'].keys())
    ner = row['entidades']
    topic = topics.get(row_topic)

    for frase in row['text'].split('.'):
        try:
            anotacion = client.annotate(frase)
            time.sleep(0.3)
            for sentence in anotacion.sentence:
                pos = devolver_pos(sentence)
                tripletas = devolver_tripletas(sentence, frase, pos, tripletgenerator)
                resultado.extend(tripletas)
        except Exception:
            continue

    return resultado
