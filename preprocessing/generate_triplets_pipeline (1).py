
import joblib
import dask.dataframe as dd
from text_pipeline.triplet_extraction import anotar_fila
from semantic_oboe.triplet_manager_lib import TripletManager
import pandas as pd

# Ruta del dataset y de salida
input_path = '../bbc_objects/bbc_processed_final_semantic_2'
output_path = '../bbc_objects/bbc_semantic_tripletas_simplificado'
triplet_csv_path = '../bbc_objects/dataset_triplet_bbc_new_simplificado.csv'

# Cargar datos
df = joblib.load(input_path)
ddf = dd.from_pandas(df, npartitions=10)

# Cargar tópicos
topics = joblib.load('/Users/Raul/doctorado/semantic_oboe/semantic_oboe/bbc_objects/new_bbc_topics_7_sinprob')
tripletmanager = TripletManager()

# Aplicar anotación y extracción de tripletas
def aplicar_tripletas(df):
    return df.apply(lambda row: anotar_fila(row, tripletmanager, topics), axis=1)

ddf['tripletas'] = ddf.map_partitions(aplicar_tripletas, meta=('object')).compute()

print('Extracción completa. Guardando resultados...')
joblib.dump(ddf, output_path)

print('Generando dataframe de tripletas...')
df = ddf.compute()
df.drop(df[df['tripletas'].map(len) == 0].index, inplace=True)

triplet_df_def = df.explode('tripletas').reset_index()
triplet_df_kge = pd.DataFrame(triplet_df_def['tripletas'].tolist(), index=triplet_df_def.index)
triplet_df_kge['new_topic'] = triplet_df_def.new_target
triplet_df_kge['old_index'] = triplet_df_def.index
triplet_df_kge.columns = ['subject', 'relation', 'object', 'new_topic', 'old_index']
triplet_df_kge.to_csv(triplet_csv_path, index=False)

print(f'Tripletas guardadas en {triplet_csv_path}')
