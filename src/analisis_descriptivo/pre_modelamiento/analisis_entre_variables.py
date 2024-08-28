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
Comparar variables categóricas con la variable por predecir
"""
for var_cat in VARS_CATEGORICAS:
  if var_cat == 'class': continue

  frecuencias = datos.groupby(['class', var_cat]).size().reset_index(name = 'Cantidad')

  # Frecuencias de 'class' en cada variable categórica diferente
  plt.figure(figsize = (12, 6))
  sns.barplot(x = 'class', y = 'Cantidad', hue = var_cat, data = frecuencias)
  plt.title(f'Frecuencias de "class", por {var_cat}')
  plt.xticks(rotation = 45)
  plt.show()

  # Frecuencias de las otras variables categóricas, en cada categoría de 'class'
  plt.figure(figsize = (12, 6))
  sns.barplot(x = var_cat, y = 'Cantidad', hue = 'class', data = frecuencias)
  plt.title(f'Frecuencias de {var_cat} por "class"')
  plt.xticks(rotation = 45)
  plt.show()

# %%
"""
Comparar variables numéricas con la variable por predecir
"""
for var_num in VARS_NUMERICAS:
  plt.figure(figsize = (12, 6))
  sns.violinplot(x = 'class', y = var_num, data = datos, inner = None)
  sns.boxplot(x = 'class', y = var_num, data = datos, width = 0.1)
  plt.title(f'Distribution of {var_num} by class')
  plt.show()

# %%
# Correlación entre las variables numéricas
matriz_de_correlaciones = datos[VARS_NUMERICAS].corr()

plt.figure(figsize = (10, 8))
sns.heatmap(
  matriz_de_correlaciones, 
  annot = True, 
  cmap = 'coolwarm', 
  vmin = -1, 
  vmax = 1, 
  fmt = '.2f',
  linewidths = 0.5
)
plt.show()

# %%
# Gráficos de dispersión entre cada par de variables numéricas
sns.pairplot(datos[VARS_NUMERICAS])
