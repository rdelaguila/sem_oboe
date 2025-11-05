import os
import json
import pandas as pd
from statistics import mean

# Ruta base de los resultados
BASE_DIR = "../data/explanations_eng"

# Diccionario donde guardaremos los resultados
results = {}

for corpus in os.listdir(BASE_DIR):
    corpus_path = os.path.join(BASE_DIR, corpus)
    if not os.path.isdir(corpus_path):
        continue

    summary_path = os.path.join(corpus_path, "all_topics_summary.json")
    if not os.path.isfile(summary_path):
        print(f"⚠️ No se encontró all_topics_summary.json en {corpus_path}")
        continue

    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        topics = data.get("topics_metrics", [])
        if not topics:
            continue

        coherence_vals = [t.get("avg_coherence", 0) for t in topics]
        relevance_vals = [t.get("avg_relevance", 0) for t in topics]
        coverage_vals = [t.get("avg_coverage", 0) for t in topics]

        results[corpus] = {
            "num_topics": len(topics),
            "coherencia_media": mean(coherence_vals),
            "relevancia_media": mean(relevance_vals),
            "cobertura_media": mean(coverage_vals)
        }

    except Exception as e:
        print(f"❌ Error procesando {summary_path}: {e}")

# Convertir resultados a DataFrame
df = pd.DataFrame([
    {
        "Corpus": corpus,
        "Nº Topics": v["num_topics"],
        "Coherencia media": v["coherencia_media"],
        "Relevancia media": v["relevancia_media"],
        "Cobertura media": v["cobertura_media"]
    }
    for corpus, v in results.items()
])

# Calcular promedios globales
global_row = {
    "Corpus": "GLOBAL",
    "Nº Topics": df["Nº Topics"].sum(),
    "Coherencia media": df["Coherencia media"].mean(),
    "Relevancia media": df["Relevancia media"].mean(),
    "Cobertura media": df["Cobertura media"].mean()
}
df = pd.concat([df, pd.DataFrame([global_row])], ignore_index=True)

# Mostrar resultados
print("\n=== Métricas medias por corpus ===")
print(df.to_string(index=False))

# Guardar resultados en CSV
output_path = "corpus_metrics_summary.csv"
df.to_csv(output_path, index=False, encoding="utf-8")
print(f"\n✅ Resultados guardados en: {output_path}")
