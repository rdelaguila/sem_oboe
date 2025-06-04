from pykeen.pipeline import pipeline
import pandas as pd
import os

TRIPLES_PATH = "data/triples_raw/triples_bbc.pkl"
KG_PATH = "data/triples_raw/triples_bbc.tsv"
EMB_DIR = "data/triples_emb/"
os.makedirs(EMB_DIR, exist_ok=True)

df = pd.read_pickle(TRIPLES_PATH)
with open(KG_PATH, "w") as fout:
    for triples in df['triples']:
        for s, r, o in triples:
            fout.write(f"{s}\t{r}\t{o}\n")

result = pipeline(
    training=KG_PATH,
    model="TransE",
    model_kwargs=dict(embedding_dim=100),
    training_kwargs=dict(num_epochs=100),
    optimizer="Adam",
    optimizer_kwargs=dict(lr=0.001),
)
result.save_to_directory(EMB_DIR)
print(f"Embeddings guardados en {EMB_DIR}")
