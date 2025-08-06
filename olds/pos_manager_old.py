import nltk
import spacy
import pandas as pd
import numpy as np
import sys
import re
import joblib


class PosManager(object):
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        self.sujeto_no_permitido = ['VB','VBN','VBP','VB','VH','CC','DT','EX','IN','RB','SYM','WRB','TO','UH','MD']
        self.sujeto_permitido = ['VBZ','NN','NNP','NNPS','NNS','FW','PRP','CD','WP']
        self.predicado_permitido = ['VB','VBN','VBP','VB','MD','VBZ','VBD']

        self.objeto_no_permitido = ['VB','VBN','VBP','VB','CC','DT','EX','IN','RB','SYM','WRB','TO','UH','MD']
        self.objeto_permitido = ['VBZ','NN','NNP','NNPS','NNS','FW','CD','WP','RBR','RBS','JJ','JJS','JJR']
        self.preposiciones  = ['in','a','at','on','onto','upon','above','below','above', 'across', 'against', 'along', 'among', 'around', 'at', 'before', 'behind', 'below', 'beneath', 'beside', 'between', 'by', 'down', 'from', 'in', 'into', 'near',
                 'of', 'off', 'on', 'to', 'toward', 'under', 'upon', 'with', 'within']
    def son_palabras_equivalentes(self,tokena,tokenb):
        """Dev"""
        if (len(tokena)==0 or len(tokenb)==0):
            return False
        if tokena.lower() == tokenb.lower():
            return True
        elif [token.lemma_ for token in self.nlp(tokena)][0].lower()==tokenb.lower() or [token.lemma_ for token in self.nlp(tokenb)][0].lower()==tokena.lower():
            return True
        else:
            return False

    def buscar_pos_palabra (self,palabra, pos: dict):
        try:

            lista = [pos.get(t) for t in pos.keys() if self.son_palabras_equivalentes(self.normalizar_token(t),self.normalizar_token(palabra))]
            if len(lista)>0:
                return lista
            elif palabra in ['am','is','are','do','have','does','has','can']: #se hace porque stanfordnlp a veces añade verbos xa encontrar sentido en tripletas. reduce precisión, pero por contra incluye más tripletas
                return 'VB'
            else:
                return '.'
        except:
            print('Excepcion en buscar_pos_palabra')
            print('palabra: '+palabra)
            print('pos '+str(pos))

    def buscar_pos(self,palabras,pos):
        #print ('POS '+str(pos))
        if len (palabras.split(' '))==1:
            return self.buscar_pos_palabra(palabras,pos)[0]
        else:
            lista_pos = []
            for token in palabras.split(' '):
                lista_pos.extend(self.buscar_pos_palabra(token,pos)) #[t[1] for t in pos if  self.son_palabras_equivalentes(t[0],token)]     t[0].replace('\n','').replace('\r','').replace('\t','')==token.replace('\n','').replace('\r','').replace('\t','').strip()
            return lista_pos

    #def pos_tag (self,frase):
    #    #print('frase en pos '+frase)
    #    pos = nltk.pos_tag(frase.replace("  "," ").strip().split(' '))

    #    return [t for t in pos if len(t[0].strip())>0 ]


    def es_frase_valida(self,pos):
        """la frase será válida si tiene al menos un sujero permitido, un predicado permitido, y un verbo permitido en su pos. Frases como the NJ PITT at 1 00 on ABC, LA CAL at 3 00 CBC , BUFF BOS at 7 00 TSN and FOX , and MON QUE at 7 30 CBC serán no válidas"""
        encontradoS = False
        encontradoP = False
        encontradoO = False

        for p in pos.keys():
            if pos.get(p) in self.sujeto_permitido:
                encontradoS = True
            elif pos.get(p) in self.objeto_permitido:
                encontradoO = True
            elif pos.get(p) in self.predicado_permitido:
                encontradoP = True
            else:
                continue
        return encontradoS and encontradoP and encontradoO

    def  resolve_contractions (self,frase):
        return frase.replace("""\'s""",' is').replace("""n't""",' not').replace("""\'ll""",""" will""").replace("""\'m'""",' am').replace("""can not""",'cannot')

    def normalizar_frase(self,frase):
        frase = self.resolve_contractions(frase)
        frase = frase.replace(',',' , ')# esto lo hago para evitar cosas como Als,it
        frase = re.sub('[^A-Za-z0-9]+', ' ', frase)
        frase = re.sub(' +', ' ', frase)
        frase = frase.strip()

        return frase

    def normalizar_token(self,token):
        token = re.sub('[^A-Za-z0-9]+', '', token)
        return token

    def devolver_pos_palabras(self, predicado, pos):
        nuevodict = dict()
        for p in predicado.split(' '):
            nuevodict[p] = pos.get(p)
        return nuevodict


class PredicateManager (PosManager):


    def __init__(self):
        super().__init__()


    def devolver_sin_modal(self,predicado):
        modal = ["can", "cannot","can't", "could" , "couldn't", "couldnt", "dare" , "may",  "might", "must", "need", "ought",  "shall", "should"
        "shouldn't", "shouldnt","should not", "shall not","will", "would"]
        return " ".join([w for w in predicado.split(' ') if not w.lower() in modal])

    def devolver_sin_adverbio(self,predicado,pos):
        if len(predicado.split(' '))>1:

            adv = [t for t in pos.keys() if pos.get(t) in ['RB','RBS','RBP']]
            if len(adv)>=1:
                predicado = predicado.replace(adv[0],'')
        return predicado

    def es_predicado_valido(self,predicado,pos):
        #sujeto_permitido = ['NN','NNP','NNPS','NNS','JJ']
        encontradoP = False
        visitadoP = False

        for w in predicado.split(' '):
            #print ('EAA '+w)
            pos_predicado = self.buscar_pos(w,pos)

            if not visitadoP and pos_predicado in self.predicado_permitido:
                encontradoP = True
                visitadoP = True

            if pos_predicado in self.sujeto_permitido:
                return False
        return encontradoP

    def es_sujeto_valido(self,predicado,pos):
        #sujeto_permitido = ['NN','NNP','NNPS','NNS','JJ']
        encontradoS = False
        visitadoS = False

        for w in predicado.split(' '):
            #print ('EAA '+w)
            pos_predicado = self.buscar_pos(w,pos)
            if not visitadoS and p in self.sujeto_permitido:
                encontradoS = True
                visitadoS = True

            if pos_predicado in self.predicado_permitido:
                return False

        return encontradoS

    def es_objeto_valido(self,predicado,pos):
        #sujeto_permitido = ['NN','NNP','NNPS','NNS','JJ']
        encontradoP = False
        visitadoP = False

        for w in predicado.split(' '):
            #print ('EAA '+w)
            pos_predicado = self.buscar_pos(w,pos)
            for p in pos_predicado:
                if not visitadoP and p in self.objeto_permitido:
                    encontradoP = True
                    visitadoP = True

                if p in self.predicado_permitido:
                    return False

        return encontradoP

    def buscar_indice_pos_palabra(self, palabra, pos_a_buscar, pos):

        cont = 0

        for p in pos.keys():
            if p == palabra:
                pospalabra = self.buscar_pos(palabra,pos)
                if pos_a_buscar == pospalabra:
                    return cont
                else:
                    return -1
            else:
                cont = cont + 1
                continue
        return -1

    def buscar_patron_VNI (self,predicado,pos):
        verbo = ['VB','VBN','VBP','VB','MD','VBZ','VBD']
        noun = ['NN','NNP','NNS','NNPS']
        indice_verbo = -1
        indice_nn = -1
        indice_in = -1
        encontrado = False
        for w in predicado.split(' '):
            minipos = self.buscar_pos(w,pos)
            if minipos in verbo:
                encontrado = True
                break
        if not encontrado:
            #print ('no encontrado')
            return False
        encontrado_w = False
        encontrado_nn = False
        for w in predicado.split(' '):
            for v in verbo:
                indice_verbo = self.buscar_indice_pos_palabra(w, v, pos)
                if indice_verbo>-1:
                    #print (indice_verbo)
                    encontrado_w = True
                    break
            if encontrado_w:
                break
        for w in predicado.split(' '):
            for n in noun:
                indice_nn = self.buscar_indice_pos_palabra(w, n, pos)
                if indice_nn>-1:
                    #print (indice_nn)
                    encontrado_nn = True
                    break
            if encontrado_nn:
                break

        for w in predicado.split(' '):
            indice_in = self.buscar_indice_pos_palabra(w, 'IN', pos)
            if (indice_in>-1):
                break
        #print ('indices v:{0}nn:{1}i:{2}'.format(indice_verbo,indice_nn,indice_in))
        if indice_in == -1 or indice_verbo == -1 or indice_nn == --1:
            return False
        else:
            return indice_verbo<indice_nn<indice_in

    def patron_verbo_preposicion (self,predicado,pos):
        verbo = ['VB','VBN','VBP','VB','MD','VBZ','VBD']

        if len (predicado.split(' '))==2:
            if self.buscar_pos(predicado.split(' ')[0],pos) in verbo and predicado.split(' ')[1] in self.preposiciones:
                return True
        elif len(predicado)==1 and predicado == 'isAt' or predicado == 'locatedAt':
            return True
        return False


    def normalizar_predicado(self,predicado,pos):
        haves = ['has','have','had']
        bes = ['is','are','am','was','were']

        if len(predicado.split(' '))>2 or self.patron_verbo_preposicion(predicado,pos)==False:
            return " ".join([w for w in predicado.split(' ') if not w.casefold().strip()  in bes and not w.casefold().strip()  in haves])
        else:
            return predicado

    def devolver_verbo_predicado_normalizado(self,predicado,pos,lemma=False):
        import nltk
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        if len(predicado.split(' '))>1:
            predicado = self.devolver_sin_modal(predicado)
            predicado = self.normalizar_predicado(predicado,pos)
            predicado = self.devolver_sin_adverbio(predicado,pos)

            if lemma:
                verbo = [t for t in pos.keys() if pos.get(t) in ['VB','VBD','VBN','VBP','VB','VBZ']]
                verbo_lema = lemmatizer.lemmatize(verbo[0],pos='v')
                return predicado.replace(verbo,verbo_lema)
            else:
                return predicado
        else:
            if lemma:
                return lemmatizer.lemmatize(predicado,pos='v')
            else:
                return predicado


