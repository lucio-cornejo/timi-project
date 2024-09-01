# %% [markdown]
# # Pre procesamiento de datos

# %%
from src.utils import (
  COLUMNA_ID,
  PREDICTORES_NUMERICOS,
  VARIABLES_CATEGORICAS
)

import pandas as pd

# %%
datos = pd.read_csv('census_income/census-income.csv')
datos

# %% [markdown]
# ## Observaciones/filas duplicadas

# En el archivo 'census_income/census-income.txt' se especifica
# que la columna que identifica a cada fila del conjunto de datos
# se denomina 'key'.

# Inspeccionamos que aquella columna no presente duplicados, pues
# eso sugeriría la presencia de observaciones repetidas

# %%
# Número de identificadores duplicados
datos[datos[COLUMNA_ID].duplicated()].shape[0]

# %% [markdown]
# Como no se hallaron duplicados en base a la columna que identifica
# cada fila, inspeccionaremos todas las columnas en búsqueda de duplicados

# %%
# Número de filas duplicadas
datos[datos.duplicated()].shape[0]

# %% [markdown]
# **Conclusión**: Los datos no presentan filas duplicadas.

# %% [markdown]
# ## Valores vacíos/perdidos

# %%
# Cantidad de valores perdidos en cada predictor numérico
for predictor_num in PREDICTORES_NUMERICOS:
  print(
    predictor_num, 
    ':\t',
    pd.to_numeric(datos[predictor_num], errors = 'coerce').isna().sum(),
    sep = ''
  )

# %% [markdown]
# **Conclusión**: No existen valores perdidos en ninguno de los predictores numéricos.

# %% [markdown]
# Note que no debería ser necesario revisar la existencia de valores
# perdidos en la columna que identifica cada fila. 
# Esto porque las llaves primarias (primary key) no admiten valores perdidos.
# 
# No obstante, verificaremos que dicha columna no presenta valores vacíos

# %%
# Como la columna identificadora es de tipo número entero,
# contabilizamos vacíos de la misma manera que con los predictores numéricos
pd.to_numeric(datos[COLUMNA_ID], errors = 'coerce').isna().sum()

# %% [markdown]
# Conclusión: La columna identificadora no presenta valores perdidos.

# %%
# Cantidades por variable categórica
for var_cat in VARIABLES_CATEGORICAS:
  print(datos[var_cat].value_counts(), '\n')

# %%
# Cantidad de valores vacíos en las columnas asociadas
# a las variables categóricas
datos[VARIABLES_CATEGORICAS].isna().sum().sum()

# %% [markdown]
# Note que, tal como se especificó en el archivo 'census+income/adult.names',
# los valores perdidos han sido reeemplazados por '?' .
# 
# Asimismo, en varias variables categóricas se tiene el valor "Not in universe".
# Tal valor se refiere a que la columna en cuestión 
# [no se aplica](https://forum.ipums.org/t/dhs-datasets-not-in-universe-and-missing-values/2575)
# para aquella persona/observación.
# 
# A continuación, contabilizaremos la cantidad de valores perdidos en las 
# variables categóricas, contando las observaciones de la forma '?' .

# %%
vacios = (datos[VARIABLES_CATEGORICAS]
  .apply(
    lambda var_cat: var_cat
      # Remover espacio en blanco antes de revisar igualdad con '?',
      # en caso por error se haya registrado algún valor, por ejemplo,
      # como '? ', ' ? ', etc.
      .apply(lambda cat: str(cat).replace(' ', '') == '?')
      .sum(),
    axis = 0
  )
  .to_frame()
  .sort_values(0, ascending = False)
  .set_axis(['Vacíos (cantidad)'], axis = 'columns')
)
vacios['Vacíos (porcentaje)'] = round(100 * vacios['Vacíos (cantidad)'] / datos.shape[0], 2)
vacios

# %% [markdown]
# Note en la tabla previa, que cuatro columnas (
# "migration code-change in msa", "migration code-change in reg", 
# "migration code-move within reg" y "migration prev res in sunbelt"
# ) presentan alrededor de 50% de datos vacíos.
# Como tal porcentaje es tan alto, esto sugiere no descartar las filas
# donde alguna de esas cuatro columnas no presenta datos, sino, preservar
# la transformación de los datos categóricos vacíos en una categoría
# nueva, representada por '?'.

# Por otro lado, el resto de columnas posee un porcentaje de vacíos
# menor a 3.5%. Aquel porcentaje es tan bajo, que, como la cantidad de
# filas es significativa (aproximadamente 200 mil), podemos considerar
# descartar las filas en las que alguna de esas columnas no presenta datos
# (es decir, es de la forma '?').

# %% [markdown]
# ## Conclusión
# 
# Preservaremos los valores perdidos (presentes solo en variables
# categóricas) vía su respresentación '?', como si se tratase de una categoría aparte.
# 
# Después de analizar y visualizar los predictores, decidiremos qué tipo de 
# transformaciones extra realizaremos a los datos, antes de su uso en el modelamiento predictivo.