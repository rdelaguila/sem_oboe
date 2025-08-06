import pandas as pd
import numpy as np
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from pykeen.evaluation import RankBasedEvaluator
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score

import os

TRIPLES_PATH = "data/triples_ft/processed/dataset_final_triplet_bbc_pykeen"
KG_PATH = "data/triples_ft/processed/triples_bbc.tsv"
EMB_DIR = "data/triples_emb/"
os.makedirs(EMB_DIR, exist_ok=True)
model = "TransE"
import joblib
df = joblib.load(TRIPLES_PATH)

# 2. prepare triples factory
triples = df[['subject', 'relation', 'object']].values
# using from_labeled_triples for pre-mapped triples; no inverse triples
tf = TriplesFactory.from_labeled_triples(
    triples,
    create_inverse_triples=False,
)

# 3. train/test split for kge
training_tf, testing_tf = tf.split([0.8, 0.2], random_state=0)

#evaluation_kwargs = dict(filtered=true)

# 4. train transe via pykeen pipeline with same hyperparameters
#result = pipeline(
#    training=training_tf,
##    testing=testing_tf,
#    model='transe',
#    model_kwargs=dict(
#        embedding_dim=150,
#        scoring_fct_norm=2,
#        regularizer='lp',
        # 'lambda' is a reserved keyword, using lambda_ instead
#        regularizer_kwargs=dict(p=3, weight=1e-5),
#    ),
#    optimizer='adam',
#    optimizer_kwargs=dict(lr=1e-3),
#    loss='negativeloglikelihood',

#    training_kwargs=dict(
#        num_epochs=200,
#        batch_size=100,
#        use_tqdm_batch=false,
#    ),
#    evaluator_kwargs=evaluation_kwargs,
#    random_seed=0,
#)

#model = result.model

# 5. Save the trained model
import torch
#result.save_to_directory('pykeen_transe_model')
model = torch.load(
    'pykeen_transe_model/trained_model.pkl',
    map_location='cpu',
    weights_only=False,      # ← importante
)

model.eval()
# 6. Evaluate performance on test set
#evaluator = RankBasedEvaluator(filtered=True)
#eval_results = evaluator.evaluate(
#    model=model,
#    mapped_triples=testing_tf.mapped_triples,
#    additional_filter_triples=[training_tf.mapped_triples],
#)
#print('Evaluation results:', eval_results)

# 7. Extract embeddings for classification
entity_to_id = tf.entity_to_id
#embeddings = model.entity_representations[0].real_val.detach().cpu().numpy()
#model = result.model

# Todas las entidades
from pykeen.triples import CoreTriplesFactory

training_tf = TriplesFactory.from_path_binary(
    'pykeen_transe_model/training_triples'
)
entity_to_id = training_tf.entity_to_id
id_to_entity = {idx: ent for ent, idx in entity_to_id.items()}

entity_embeddings = model.entity_representations[0](indices=None)
entity_embeddings = entity_embeddings.detach().cpu().numpy()

# Todas las relaciones
relation_embeddings = model.relation_representations[0](indices=None)
relation_embeddings = relation_embeddings.detach().cpu().numpy()
def get_embedding(entity: str) -> np.ndarray:
    idx = entity_to_id.get(entity)
    return entity_embeddings[idx] if idx is not None else np.zeros(entity_embeddings.shape[1])

# Apply to subjects and objects
subject_embs = np.vstack(df['subject'].apply(get_embedding).values)
object_embs  = np.vstack(df['object'].apply(get_embedding).values)
X = np.hstack([subject_embs, object_embs])
y = df['new_topic'].astype(int).values  # or str

# 8. Classification train/test split and model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)
clf = XGBClassifier(
    n_estimators=200,
    learning_rate=0.01,
    max_depth=6,
    use_label_encoder=False,
    eval_metric='mlogloss'
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# 9. Classification performance
print('Accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
