import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP
from scipy.stats import spearmanr

# ============================================
# CONFIGURACIÓN GENERAL
# ============================================
DATASETS = ["reuters_activities", "bbc", "amazon"]
MODEL = "transe"
BASE_PATH = "../data/model_output_081125"
EMB_PATH = "../data/triples_emb_081125"
FIG_PATH = "./figures_mdpi"
os.makedirs(FIG_PATH, exist_ok=True)

sns.set(style="whitegrid", font_scale=1.2)


# ============================================
# UTILIDAD: Cargar el JSON más reciente
# ============================================
def load_latest_results(dataset, model):
    folder = os.path.join(BASE_PATH, f"{dataset}_ces_{model}")
    if not os.path.exists(folder):
        raise FileNotFoundError(f"❌ Folder not found: {folder}")

    json_files = [
        f for f in os.listdir(folder)
        if f.startswith("classification_results_") and f.endswith(".json")
    ]
    if not json_files:
        raise FileNotFoundError(f"❌ No classification_results_*.json files in {folder}")

    json_files = sorted(json_files, key=lambda x: os.path.getmtime(os.path.join(folder, x)), reverse=True)
    latest_file = json_files[0]
    latest_path = os.path.join(folder, latest_file)

    print(f"📂 Loading latest results for {dataset}: {latest_file}")
    with open(latest_path, "r") as f:
        data = json.load(f)
    return data


# ============================================
# FIGURE 5 – Boxplots de varianza entre folds
# ============================================
def plot_boxplots_variance(datasets, model):
    metrics = []

    # mapeo fijo para que no haya líos de mayúsculas
    name_map = {
        "amazon": "Amazon",
        "bbc": "BBC",
        "reuters_activities": "Reuters",
    }

    for ds in datasets:
        res = load_latest_results(ds, model)
        cv = res.get("cv_results", {})
        folds = len(cv.get("fold_aucs", []))

        label = name_map.get(ds, ds)  # por si algún día metes otro

        for i in range(folds):
            metrics.append({
                "Dataset": label,
                "Fold": i + 1,
                "AUC": cv["fold_aucs"][i],
                "LogLoss": cv["fold_logloss"][i],
                "MPCE": cv["fold_mean_per_class_error"][i]
            })

    df = pd.DataFrame(metrics)

    # orden deseado
    order = ["Amazon", "BBC", "Reuters"]
    df["Dataset"] = pd.Categorical(df["Dataset"], categories=order, ordered=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    sns.boxplot(x="Dataset", y="AUC", data=df, ax=axes[0], order=order, palette="Set2")
    axes[0].set_title("Cross-Validation AUC Variance")

    sns.boxplot(x="Dataset", y="LogLoss", data=df, ax=axes[1], order=order, palette="Set2")
    axes[1].set_title("Cross-Validation LogLoss Variance")

    sns.boxplot(x="Dataset", y="MPCE", data=df, ax=axes[2], order=order, palette="Set2")
    axes[2].set_title("Cross-Validation Mean Per Class Error")

    for ax in axes:
        ax.set_xlabel("")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_PATH, "figure5_cv_variance_boxplots.png"), dpi=300)
    plt.close()
    print("✅ Figure 5 saved: figure5_cv_variance_boxplots.png")


# ============================================
# FIGURE E.1 – MRR vs AUC Correlation
# ============================================
def plot_mrr_vs_auc(datasets, model):
    corr_data = []
    for ds in datasets:
        res = load_latest_results(ds, model)
        mrr = res["kge_metrics"]["pre_ces"]["test"]["mrr"]
        auc = res["cv_results"]["mean_auc"]

        # Renombrar "reuters_activities" a "Reuters"
        label = "Reuters" if ds == "reuters_activities" else ds.capitalize()

        corr_data.append({"Dataset": label, "KGE_MRR": mrr, "Downstream_AUC": auc})

    df = pd.DataFrame(corr_data)
    rho, p = spearmanr(df["KGE_MRR"], df["Downstream_AUC"])
    plt.figure(figsize=(6, 5))
    sns.regplot(x="KGE_MRR", y="Downstream_AUC", data=df, scatter_kws={"s": 80}, color="royalblue")
    plt.title(f"Figure: Correlation between KGE MRR and Downstream AUC\nρ={rho:.3f}, p={p:.3f}")
    for i, row in df.iterrows():
        plt.text(row["KGE_MRR"] + 0.005, row["Downstream_AUC"], row["Dataset"], fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_PATH, "figureE1_mrr_auc_correlation.png"), dpi=300)
    plt.close()
    print("✅ Figure E.1 saved: figureE1_mrr_auc_correlation.png")


# ============================================
# FIGURE E.2 – UMAP de Embeddings TransE
# ============================================
def plot_umap_embeddings_with_topics(dataset, base_path="../data", model="transe"):
    emb_dir = os.path.join(base_path, f"triples_emb/{dataset}_ces_{model}")
    emb_file = os.path.join(emb_dir, "embeddings_postces.npy")
    if not os.path.exists(emb_file):
        emb_file = os.path.join(emb_dir, "embeddings_preces.npy")
    embeddings = np.load(emb_file)
    print(f"Loaded embeddings shape: {embeddings.shape}")

    csv_path = os.path.join(base_path, f"triples_raw/{dataset}/dataset_triplet_{dataset}_new_simplificado.csv")
    df = pd.read_csv(csv_path)
    topics = df["topic"].astype(str).values
    print(f"Loaded {len(topics)} topic labels")

    n_emb = len(embeddings)
    n_topics = len(topics)
    n_min = min(n_emb, n_topics)

    if n_emb != n_topics:
        print(f"⚠️ Size mismatch: {n_emb} embeddings vs {n_topics} topics → trimming to {n_min}")
        embeddings = embeddings[:n_min]
        topics = topics[:n_min]

    unique_topics = sorted(np.unique(topics))
    print(f"Detected {len(unique_topics)} unique topics")

    reducer = UMAP(n_neighbors=100, min_dist=0.35, metric="cosine", random_state=42)
    proj = reducer.fit_transform(embeddings)

    # --- Cambiar nombre visual de Reuters ---
    display_name = "Reuters" if dataset == "reuters_activities" else dataset.capitalize()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=proj[:, 0], y=proj[:, 1],
        hue=topics,
        palette="tab10" if len(unique_topics) <= 10 else "Spectral",
        s=25, alpha=0.8, edgecolor="none"
    )
    plt.title(f"Figure E.2. UMAP projection of TransE embeddings by topic ({display_name})")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend(title="Topics", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()

    output_path = os.path.join("./figures_mdpi", f"figureE2_umap_{display_name.lower()}_topics.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Figure E.2 (by topic) saved: {output_path}")


# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    print("Generating figures for datasets:", DATASETS)
    plot_boxplots_variance(DATASETS, MODEL)
    #plot_mrr_vs_auc(DATASETS, MODEL)
    #for ds in DATASETS:
    #    plot_umap_embeddings_with_topics(ds)
    print("\n🎉 All figures generated successfully in:", FIG_PATH)
