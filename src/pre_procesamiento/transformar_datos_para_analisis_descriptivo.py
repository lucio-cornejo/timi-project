import pandas as pd


VARS_NUMERICAS = [
  'age',
  "fnlwgt",
  "education-num",
  "capital-gain",
  "capital-loss",
  "hours-per-week"
]

VARS_CATEGORICAS = [
  'workclass',         
  'education',         
  'marital-status',
  'occupation',        
  'relationship',      
  'race',              
  'sex',               
  'native-country',
  'class'
]

def etiquetar_vacios_en_train_y_test() -> tuple[pd.DataFrame]:
  datos_train = pd.read_csv('data/datos_train_limpios_con_vacios.csv', encoding = 'utf-8')
  datos_test = pd.read_csv('data/datos_test_limpios_con_vacios.csv', encoding = 'utf-8')

  # Reconstruir el conjunto de datos
  datos = pd.concat([datos_train, datos_test], ignore_index = True)

  ETIQUETA_CATEGORIA_PERDIDA = '?'
  for var_cat in VARS_CATEGORICAS:
    datos[var_cat] = pd.Categorical(datos[var_cat].fillna(ETIQUETA_CATEGORIA_PERDIDA))

  # Separar observaciones en entrenamiento y test
  return datos.iloc[0:datos_train.shape[0]], datos.iloc[datos_train.shape[0]:]
