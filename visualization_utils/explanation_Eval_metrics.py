import os
import json
import pandas as pd
from statistics import mean

BASE_DIR = "../data/explanations_eng"

results = {}

for corpus in os.listdir(BASE_DIR):
    corpus_path = os.path.join(BASE_DIR, corpus)
    if not os.path.isdir(corpus_path):
        continue

    coherences, relevances, coverages = [], [], []
    topics_with_eval = 0  # <-- contador correcto

    # recorrer subcarpetas (topics)
    for entry in os.listdir(corpus_path):
        topic_path = os.path.join(corpus_path, entry)
        if not os.path.isdir(topic_path):
            continue

        eval_path = os.path.join(topic_path, "evaluations.json")
        if not os.path.isfile(eval_path):
            continue  # no es un topic válido

        topics_with_eval += 1  # contamos solo los topics que sí tienen evaluations.json

        try:
            with open(eval_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # data es un dict: "0", "1", ...
            for _, val in data.items():
                if not isinstance(val, dict):
                    continue
                if "coherence" in val:
                    coherences.append(val["coherence"])
                if "relevance" in val:
                    relevances.append(val["relevance"])
                if "coverage" in val:
                    coverages.append(val["coverage"])

        except Exception as e:
            print(f"⚠️ Error leyendo {eval_path}: {e}")

    # guardar resultados del corpus
    if topics_with_eval > 0 and coherences:
        results[corpus] = {
            "num_topics": topics_with_eval,
            "coherencia_media": mean(coherences),
            "relevancia_media": mean(relevances),
            "cobertura_media": mean(coverages),
        }
    else:
        results[corpus] = {
            "num_topics": 0,
            "coherencia_media": None,
            "relevancia_media": None,
            "cobertura_media": None,
        }

# pasar a DataFrame
df = pd.DataFrame(
    [
        {
            "Corpus": corpus,
            "Nº Topics": v["num_topics"],
            "Coherencia media": v["coherencia_media"],
            "Relevancia media": v["relevancia_media"],
            "Cobertura media": v["cobertura_media"],
        }
        for corpus, v in results.items()
    ]
)

# fila global (solo sobre los que tienen datos)
df_valid = df[df["Nº Topics"] > 0]
global_row = {
    "Corpus": "GLOBAL",
    "Nº Topics": df_valid["Nº Topics"].sum(),
    "Coherencia media": df_valid["Coherencia media"].mean(),
    "Relevancia media": df_valid["Relevancia media"].mean(),
    "Cobertura media": df_valid["Cobertura media"].mean(),
}
df = pd.concat([df, pd.DataFrame([global_row])], ignore_index=True)

print("\n=== Métricas medias por corpus (evaluations.json) ===")
print(df.to_string(index=False))

output_path = "evaluations_summary.csv"
df.to_csv(output_path, index=False, encoding="utf-8")
print(f"\n✅ Resultados guardados en: {output_path}")
