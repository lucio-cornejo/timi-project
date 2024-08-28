# %%
from src.pre_procesamiento.transformar_datos_para_analisis_descriptivo import (
  VARS_NUMERICAS,
  VARS_CATEGORICAS,
  etiquetar_vacios_en_train_y_test
)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %%
# Cargar datos de entrenamiento y de test, donde los vacíos de las
# variables categóricas han sido reemplazados por '?'
datos_train, datos_test = etiquetar_vacios_en_train_y_test()

# %%
datos_train.info()

# %%
datos_test.info()

# %%
# Reconstruir el conjunto de datos
datos = pd.concat([datos_train, datos_test], ignore_index = True)
datos.info()

# %%
"""
Distribución/porcentajes de cada variable categórica
"""
num_observaciones = datos.shape[0]
for var_cat in VARS_CATEGORICAS:
  porcentajes_var_cat = (datos
    .groupby([var_cat]).size()
    .reset_index(name = 'cantidad')
    .sort_values(by = 'cantidad', ascending = False)
    .assign(Porcentaje = lambda d: 100 * (d['cantidad'] / num_observaciones))
  )

  plt.figure(figsize = (12, 6))
  sns.barplot(
    x = var_cat, y = 'Porcentaje',
    data = porcentajes_var_cat,
    order = porcentajes_var_cat[var_cat]
  )
  plt.xticks(rotation = 45)
  plt.show()
