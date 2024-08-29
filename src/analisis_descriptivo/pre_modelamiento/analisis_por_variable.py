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

# %%
"""
Histograma y densidad aproximada de cada variable numérica
"""
for var_num in VARS_NUMERICAS:
  sns.histplot(
    datos[var_num], 
    # También estimar la densidad de la variable numérica
    kde = True
  )
  plt.ylabel('Frecuencia')
  plt.show()

# %%
"""
Comparamos la homogeneidad de las variables numéricas
"""
(datos[VARS_NUMERICAS]
  .describe().T
  # Calcular el coeficiente de variación (C.V) de cada variable numérica
  .assign(coef_variacion = lambda d: d['std'] / d['mean'])
  # Ordenar las variables numéricas de menor a mayor C.V,
  # es decir, de mayor homogeneidad a menor .
  .sort_values('coef_variacion', ascending = True)
  .loc[:, [
    'coef_variacion', 'mean', 'std',
    'min', '25%', '50%', '75%', 'max'
  ]]
)

# %%
"""
En la tabla resultante, solo la variable numérica "education-num"
posee C.V menor o igual que 30%. 
Así, solo aquella variable numérica es homogénea, es decir,
su promedio es representativo de dicha variable.

Por otro lado, el resto de variables numéricas poseen un C.V mayor
que 30%, así que el promedio de cada una no la representa adecuadamente,
por lo que consistirían de variables heterogéneas (no homogéneas).
"""
