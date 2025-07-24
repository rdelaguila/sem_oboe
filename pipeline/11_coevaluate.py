

import os
import re
import json
import pandas as pd
import numpy as np
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

# ======== OWN CONFIG ========

OUTPUT_DIR        = 'output'
# mT5 evaluación
eval_model_name   = 'google/mt5-small'
TOPIC_ID          = 3                       # ID del tópico a procesar

import joblib

clusters = joblib.load(OUTPUT_DIR+'/explanations.json')
# 6. Texto resumen
lines = [
    f"Para el tópico {TOPIC_ID} se generaron {best_k} temas tras clustering y se evaluaron las explicaciones con mT5."
]
for cid, terms_c in clusters.items():
    lines.append(f"Tema {cid}: {', '.join(terms_c)}")
with open(os.path.join(OUTPUT_DIR,'summary.txt'),'w',encoding='utf-8') as f:
    f.write("\n".join(lines))

print(f"Pipeline completado. Salida en {OUTPUT_DIR}")
