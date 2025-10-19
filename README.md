# oBOE: Ontology-Based Explanation Engine

## Overview

OBOE (explanatiOns Based On concEpts) is a customizable framework for (semi) supervised text classification (and derived) tasks.


In this scenario, the framework is tailored for concept maps represented as KG classification and domain explanations

It combines semantic enrichment, knowledge graph embedding (KGE), and interpretable machine learning to extract meaningful explanations from topic clusters in large document collections.

The framework integrates four main components (A-D) that work sequentially to process raw documents, extract semantic information, generate knowledge graphs, and produce explainable topic-based classifications.

## Architecture

### Execution Order: A → B → C → D

```
┌─────────────────────────────────────────────────────────────────┐
│                  A: PREPROCESSING & ENRICHMENT                  │
│  (Coreference Resolution, NER, DBpedia Entity Recognition)      │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│            B: TOPIC MODELING & TRIPLET GENERATION               │
│  (LDA Topic Modeling, Triplet Extraction, Triplet Refinement)   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│          C: KNOWLEDGE GRAPH EMBEDDINGS & CLASSIFICATION         │
│  (KGE Model Training, Graph-Based Topic Classification)         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              D: EXPLANATION GENERATION & EVALUATION             │
│  (Chain-of-Thought Generation, XAI-based Evaluation, Clustering)│
└─────────────────────────────────────────────────────────────────┘
```

## Components

### Component A: Preprocessing & Semantic Enrichment
**File:** `A_preprocessing.py`

Handles initial data preparation and semantic enrichment:
- Coreference resolution (using spaCy + coreferee)
- Text cleaning and normalization
- Named Entity Recognition (spaCy NER)
- DBpedia entity linking (via Spotlight)
- Ontological enrichment (DBpedia types and SUMO normalization)

**Input:** Raw text documents (CSV format)
**Output:** Semantically enriched DataFrame with entities, types, and normalized metadata

### Component B: Topic Modeling & Triplet Generation
**File:** `B_Rordering.py` (Topic Modeling) + `A_B_2_triplet_gen.py` (Triplet Generation)

Performs topic extraction and knowledge graph triplet generation:
- LDA-based topic modeling with hyperparameter optimization
- Coherence-based model selection
- Semantic triplet extraction from text
- Filtering and validation of triplets using NER/DBpedia entities
- Topic-annotated triplet datasets generation

**Input:** Processed semantic data from Component A
**Output:** 
- LDA model with optimal parameters
- CSV files with subject-relation-object triplets
- Topic distribution matrices

### Component C: Knowledge Graph Embeddings & Classification
**File:** `C_classification_2.py`

Implements a two-phase pipeline:
- **Phase 1:** Knowledge Graph Embedding (KGE) training using PyKEEN
  - Supports multiple models: TransE, ComplEx, DistMult, ConvKB, RotatE
  - Entity and relation embedding generation
  
- **Phase 2:** Topic classification using KGE embeddings
  - XGBoost classification with hyperparameter optimization
  - Binary and multiclass classification support
  - Comprehensive evaluation metrics (Accuracy, AUC, Log Loss, etc.)

**Input:** Topic-annotated triplets from Component B
**Output:**
- Trained KGE model and embeddings
- Trained XGBoost classifier
- Classification metrics and predictions

### Component D: Explanation Generation & Evaluation
**File:** `D_explainations.py` (Spanish) / `D_explainations_eng.py` (English)

Generates interpretable explanations for topic clusters using advanced prompting:
- Hierarchical clustering with semantic similarity
- Chain-of-Thought explanation generation
- Key phrase extraction and hallucination verification
- XAI-based evaluation with specific criteria
- Silhouette-based cluster quality assessment

**Input:** LDA model and triplet data from Component B
**Output:**
- Cluster assignments and visualizations
- Generated explanations with reasoning
- XAI evaluation scores (Coherence, Relevance, Coverage)
- Executive summary reports

## Installation

### Dependencies

```bash
pip install pandas numpy scikit-learn spacy coreferee gensim xgboost torch
pip install transformers pykeen spotlight dask joblib nltk
pip install matplotlib scipy

# Download spaCy model
python -m spacy download en_core_web_lg

# Download NLTK data
python -m nltk.downloader wordnet
```

### Required External Services

1. **DBpedia Spotlight** (for entity linking)
   ```bash
   docker run -it -p 2222:80 dbpedia/spotlight spotlight.sh en
   ```

2. **CoreNLP Server** (for triplet extraction)
   ```bash
   wget http://nlp.stanford.edu/software/stanford-corenlp-4.5.2.zip
   unzip stanford-corenlp-4.5.2.zip
   cd stanford-corenlp-4.5.2
   java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000
   ```

## Usage

### Step 1: Preprocessing (Component A)

```bash
python A_preprocessing.py \
  --input_file data/raw/documents.csv \
  --output_file data/processed/documents_semantic.pkl \
  --spotlight_endpoint http://localhost:2222/rest/annotate \
  --input_column text \
  --encoding utf-8
```

**Parameters:**
- `--input_file`: Path to raw CSV document file
- `--output_file`: Output pickle file with enriched data
- `--spotlight_endpoint`: DBpedia Spotlight endpoint
- `--dbo_ontology`: Path to DBpedia ontology file
- `--sumo_ontology`: Path to SUMO ontology file
- `--batch_size`: Optional batch processing size

### Step 2: Topic Modeling & Triplet Generation (Component B)

#### 2a. Topic Modeling
```bash
python B_Rordering.py \
  --classifier xgboost \
  --raw_path data/processed/documents_semantic.pkl \
  --eval_dir data/lda_eval/amazon \
  --max_topics 15
```

#### 2b. Triplet Generation
```bash
python A_B_2_triplet_gen.py \
  --input_data data/lda_eval/amazon/df_topic.pkl \
  --output_dir data/triples_raw/amazon \
  --output_name amazon_triplets \
  --corenlp_endpoint http://0.0.0.0:9000 \
  --mode triplets_with_topics \
  --enable_filtering \
  --filter_method both
```

**Modes:**
- `triplets_only`: Generate only triplets
- `triplets_with_topics`: Include topic information
- `add_topics_to_existing`: Add topics to existing triplets
- `explode_triplets`: Generate CSV from existing data

### Step 3: Knowledge Graph Embeddings & Classification (Component C)

```bash
python C_classification_2.py
```

The script will prompt you to:
1. Select KGE model (TransE, ComplEx, DistMult, ConvKB, RotatE)
2. Specify triplet file path
3. Specify embedding directory
4. Specify output directory

Or run directly:
```bash
python C_classification_2.py \
  --kg_path data/triples_raw/amazon/triplets.csv \
  --emb_dir data/embeddings/amazon \
  --model_type transe
```

### Step 4: Explanation Generation & Evaluation (Component D)

```bash
python D_explainations_eng.py
```

**Interactive prompts:**
1. Enter repository/dataset name (e.g., "amazon", "bbc")
2. Select topic to analyze
3. Choose vocabulary source (LDA, NER, DBpedia, or combined)
4. Specify number of clusters (or use automatic optimization)

**Output files generated:**
- `clusters.json`: Cluster assignments and top terms
- `explanations.json`: Generated explanations with reasoning
- `evaluations.json`: XAI-based evaluation scores
- `silhouette_evolution.png`: Clustering quality visualization
- `dendrogram_with_cut.png`: Hierarchical clustering dendrogram
- `summary.txt`: Executive summary report

## Output Files Structure

```
data/
├── processed/
│   └── {repo}/
│       └── {repo}_processed_semantic.pkl
├── lda_eval/
│   └── {repo}/
│       ├── lda_model
│       ├── df_topic.pkl
│       ├── top_terms_by_topic.pkl
│       └── lda_coherence_grid.csv
├── triples_raw/
│   └── {repo}/
│       └── dataset_triplet_{repo}.csv
├── triples_emb/
│   └── {repo}/
│       ├── {model}_model.pkl
│       └── {model}_mappings.pkl
├── model_output/
│   └── {repo}/
│       ├── kge_training_results_*.json
│       ├── topic_classification_results_*.json
│       └── predictions_xgboost.csv
└── explainations/
    └── {repo}/
        ├── clusters.json
        ├── explanations.json
        ├── evaluations.json
        ├── summary.txt
        ├── silhouette_evolution.png
        └── dendrogram_with_cut.png
```

## Key Features

### Semantic Enrichment
- Multi-level entity recognition (NER + DBpedia)
- Ontological type enrichment
- Coreference resolution for improved entity linking

### Topic Modeling
- Automatic optimal topic number detection
- Hyperparameter grid search (Alpha, Beta)
- Coherence-based model evaluation
- Interactive model selection

### Knowledge Graph Analysis
- Multiple KGE models for flexible graph representation
- Semantic embedding-based classification
- High-dimensional feature extraction

### Explainability
- Chain-of-Thought explanation generation
- Hallucination verification mechanisms
- XAI-based evaluation framework
- Silhouette-based cluster quality metrics

### Quality Metrics
- **Coherence:** Semantic relationship between cluster terms
- **Relevance:** Domain-specific term relationships
- **Coverage:** Completeness of explanation
- **Silhouette Score:** Clustering quality indicator

## Configuration Files

### Ontology Files Required
- `dbpedia_2016-10.owl`: DBpedia ontology (downloadable from DBpedia)
- `SUMO.owl`: SUMO upper ontology (downloadable from SUMO project)

## Troubleshooting

### CoreNLP Connection Issues
Ensure CoreNLP is running on port 9000:
```bash
telnet localhost 9000
```

### Spotlight Entity Linking Failure
Check Spotlight is accessible:
```bash
curl "http://localhost:2222/rest/annotate?text=Barack%20Obama&confidence=0.5"
```

### Out of Memory Errors
Reduce batch size in preprocessing or increase system RAM allocation

### GPU-related Issues
Set `CUDA_VISIBLE_DEVICES` to specify GPU usage or remove CUDA dependencies for CPU-only execution


## License

MIT License - See LICENSE file for details

## Contact

For issues, questions, or contributions, please open an issue on the project repository.

## Acknowledgments

- DBpedia project for entity linking
- PyKEEN for knowledge graph embedding implementations
- Stanford CoreNLP for information extraction
- spaCy and Hugging Face for NLP models