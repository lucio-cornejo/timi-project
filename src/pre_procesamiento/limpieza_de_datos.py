# %%
from src.pre_procesamiento.carga_de_datos import cargar_datos_train_test

import pandas as pd

# %%
datos_train, datos_test = cargar_datos_train_test()
print(datos_train.shape, datos_test.shape)

# %%
datos = pd.concat([datos_train, datos_test], ignore_index = True)
datos

# %%
datos.dtypes

# %%
"""
Como los valores númericos han sido cargados correctamente,
nos enfocamos en las variables categóricas.

Esto, teniendo como referencia su descripción en
archivo 'census+income/adult.names' 
"""
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


for var_cat in VARS_CATEGORICAS:
  print(datos[var_cat].value_counts(), '\n')

# %%
"""
Del resultado previo, notamos que, como se especifica en el archivo
'census+income/adult.names', los valores perdidos se han reemplazado
por '?'.

Además, la variable categórica 'class' consiste de cuatro valores 
únicos, "<=50K", "<=50K.", ">50K", ">50K.", 
pese a que tal variable se definió como una variable binaria.
"""
# Reemplazamos '?' por valor vacío
for var_cat in VARS_CATEGORICAS:
  datos.loc[:, var_cat] = datos[var_cat].replace({'?' : None})

# Establecemos dos valores únicos para la variable 'class'
datos.loc[:, 'class'] = datos['class'].replace({
  '<=50K.' : '<=50K',  
  '>50K.' : '>50K'
})

datos['class'].value_counts()

# %%
"""
Valores perdidos
"""
vacios = (datos
  .isna().sum()
  .to_frame()
  .sort_values(0, ascending = True)
  .set_axis(['vacios'], axis = 'columns')
)
vacios['vacios%'] = round(100 * vacios['vacios'] / datos.shape[0], 2)
vacios

# %%
"""
Solo las variables 'native-country', 'workclass' y 'occupation'
presenetan valores perdidos. Además, su porcentaje de valores
perdidos es pequeño, menor incluso que 6% .
"""

# %%
"""
Para el análisis descriptivo, guardamos los datos limpios, 
con vacíos, tanto de entrenamiento como de test
"""
datos_train_limpios_con_vacios = datos.iloc[0:datos_train.shape[0]]
print(datos_train_limpios_con_vacios.shape[0])
print(datos_train.shape[0])

datos_train_limpios_con_vacios.to_csv(
  'data/datos_train_limpios_con_vacios.csv',
  index = False
)

# %%
datos_test_limpios_con_vacios = datos.iloc[datos_train.shape[0]:]
print(datos_test_limpios_con_vacios.shape[0])
print(datos_test.shape[0])

datos_test_limpios_con_vacios.to_csv(
  'data/datos_test_limpios_con_vacios.csv',
  index = False
)
