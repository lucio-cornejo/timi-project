from src.pre_procesamiento.carga_de_datos import cargar_datos_train_test

import pandas as pd


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


def filtrar_datos_train_test_para_modelos() -> tuple[pd.DataFrame]:
  datos_train, datos_test = cargar_datos_train_test()

  datos = pd.concat([datos_train, datos_test], ignore_index = True)

  # Reemplazamos '?' por valor vacío
  for var_cat in VARS_CATEGORICAS:
    datos.loc[:, var_cat] = datos[var_cat].replace({'?' : None})

  # Establecemos dos valores únicos para la variable 'class'
  datos.loc[:, 'class'] = datos['class'].replace({
    '<=50K.' : '<=50K',  
    '>50K.' : '>50K'
  })

  # Como los porcentajes de valores vacíos son pequeños, y la cantidad
  # de observaciones de entrenamiento y test son suficientemente grandes,
  # descartaremos las filas que contengan al menos un valor perdido/vacío
  datos_train_para_modelos = datos.iloc[0:datos_train.shape[0]].dropna(axis = 0, how = 'any')
  datos_test_para_modelos = datos.iloc[datos_train.shape[0]:].dropna(axis = 0, how = 'any')

  return datos_train_para_modelos, datos_test_para_modelos
