import os
import json
import pandas as pd
from statistics import mean

# Ruta base donde se encuentran los datos
BASE_DIR = "../data/explanations_eng"

results = {}

for corpus in os.listdir(BASE_DIR):
    corpus_path = os.path.join(BASE_DIR, corpus)
    if not os.path.isdir(corpus_path):
        continue

    silhouette_values = []

    for topic in os.listdir(corpus_path):
        topic_path = os.path.join(corpus_path, topic)
        if not os.path.isdir(topic_path):
            continue

        clusters_path = os.path.join(topic_path, "clusters.json")
        if not os.path.exists(clusters_path):
            continue

        try:
            with open(clusters_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "global_sil" in data:
                silhouette_values.append(data["global_sil"])
        except Exception as e:
            print(f"⚠️ Error leyendo {clusters_path}: {e}")

    if silhouette_values:
        results[corpus] = {
            "num_topics": len(silhouette_values),
            "silhouette_medio": mean(silhouette_values)
        }
    else:
        results[corpus] = {
            "num_topics": 0,
            "silhouette_medio": None
        }

# Convertimos resultados a DataFrame
df = pd.DataFrame([
    {"Corpus": c, "Nº Topics": v["num_topics"], "Silhouette medio": v["silhouette_medio"]}
    for c, v in results.items()
])

# Calculamos promedio global
global_mean = df["Silhouette medio"].dropna().mean()
print("\n=== Silhouette medio por corpus ===")
print(df.to_string(index=False))
print(f"\n🌍 Silhouette medio global: {global_mean:.4f}")

# Guardamos a CSV
output_path = "silhouette_summary.csv"
df.to_csv(output_path, index=False, encoding="utf-8")
print(f"\n✅ Resultados guardados en: {output_path}")
