import pandas as pd
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import os

MODEL_PATH = "models/lora_trex"
EXPL_PATH = "data/explicaciones/explicaciones_bbc.pkl"
OUT_PATH = "data/explicaciones/coevaluacion_bbc.pkl"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, load_in_4bit=True, device_map="auto")
coevaluator = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=32, device=0)

df_exp = pd.read_pickle(EXPL_PATH)
coevaluaciones = []
for idx, row in df_exp.iterrows():
    prompt = (
        f"Dada la explicación generada para la terna {row['terna']} en el cluster {row['cluster']} "
        f"(palabras clave: {', '.join(row['palabras_cluster'])}, Silhouette: {row['silhouette']}), "
        f"valora la calidad de la explicación. Responde SOLO con: 'BUENA', 'REGULAR' o 'POBRE'.\n\n"
        f"Explicación:\n{row['explicacion']}"
    )
    rating = coevaluator(prompt)[0]['generated_text'].split('\n')[0]
    coevaluaciones.append(rating.strip())

df_exp['coevaluacion'] = coevaluaciones
df_exp.to_pickle(OUT_PATH)
print(f"Guardadas {len(coevaluaciones)} coevaluaciones en {OUT_PATH}")
