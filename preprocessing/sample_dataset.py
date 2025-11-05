import json
import pandas as pd
import numpy as np
import string

# ========================
# Configuración
# ========================
input_file = "../data/corpus_raw/arxiv/arxiv-metadata-oai.json"  # JSON Lines original
output_file = "../data/corpus_raw/arxiv/arxiv_subset_5000.csv"
max_classes = 10
n_samples = 5000
random_seed = 42


# ========================
# Cargar dataset
# ========================
data = []
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            obj = json.loads(line)

            # descartamos si no trae abstract o versions
            if 'abstract' not in obj or 'versions' not in obj or len(obj['versions']) == 0:
                continue

            # extraer categories de la primera versión
            version0 = obj['versions'][0]
            category =obj['categories'].split(' ')[0]
            if not category:
                continue

            # ✅ mantener solo documentos con **una sola categoría**
            if " " in category.strip():  # si hay espacios, hay varias => saltar
                continue

            data.append({
                'abstract': obj['abstract'],
                'category': category.strip()
            })

        except json.JSONDecodeError:
            continue

df = pd.DataFrame(data)
print(f"Documentos con UNA categoría encontrados: {len(df)}")

# ========================
# Seleccionar hasta 10 clases
# ========================
top_classes = df['category'].value_counts().nlargest(max_classes).index.tolist()
df_filtered = df[df['category'].isin(top_classes)]

# ========================
# Asignar target numérico
# ========================
class_to_idx = {cls: idx for idx, cls in enumerate(top_classes)}
df_filtered['target'] = df_filtered['category'].map(class_to_idx)

# ========================
# Muestreo proporcional
# ========================
class_counts = df_filtered['category'].value_counts()
total = class_counts.sum()
samples_per_class = (class_counts / total * n_samples).round().astype(int)

# ajuste si suma ≠ 5000
diff = n_samples - samples_per_class.sum()
if diff != 0:
    largest = samples_per_class.idxmax()
    samples_per_class[largest] += diff

np.random.seed(random_seed)
dfs = []

for cls, n_cls in samples_per_class.items():
    cdf = df_filtered[df_filtered['category'] == cls]
    n_take = min(n_cls, len(cdf))
    dfs.append(cdf.sample(n=n_take, random_state=random_seed))

df_subset = pd.concat(dfs).sample(frac=1, random_state=random_seed).reset_index(drop=True)

# ========================
# Limpiar texto
# ========================
translator = str.maketrans('', '', string.punctuation)
df_subset['text'] = df_subset['abstract'].apply(lambda x: x.translate(translator))

# ========================
# Guardar CSV final
# ========================
df_final = df_subset[['text', 'target']]
df_final.to_csv(output_file, sep=';', index=False, quoting=1)  # quoting=1 => siempre entre comillas

print(f"\n✅ CSV guardado en: {output_file}")
print(f"🧾 Documentos finales: {len(df_final)}")
print("\n📊 Mapeo de categorías → target:")
print(class_to_idx)
