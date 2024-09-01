# %% [markdown]
# # Análisis descriptivo

# %%
from src.utils import (
  COLUMNA_OBJETIVO,
  PREDICTORES_NUMERICOS,
  PREDICTORES_CATEGORICOS,
  VARIABLES_CATEGORICAS
)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %%
datos = (pd
  .read_csv('census_income/census-income.csv')
  # Descartar la columna de identificador de filas (primary key),
  # para solo analizar los predictors y la variable objectivo
  [[*PREDICTORES_NUMERICOS, *PREDICTORES_CATEGORICOS, COLUMNA_OBJETIVO]]
)

# Corregir los tipos de variables
for var_numerica in PREDICTORES_NUMERICOS:
  datos[var_numerica] = pd.to_numeric(datos[var_numerica], errors = 'coerce')

for var_categorica in VARIABLES_CATEGORICAS:
  datos[var_categorica] = pd.Categorical(datos[var_categorica])

datos

# %% [markdown]
# ## Análisis por variable

# %% [markdown]
# ### Variables categóricas

# %%
# Número de predictores categóricos
len(PREDICTORES_CATEGORICOS)

# %%
# Porcentajes de cada predictor categórico
num_observaciones = datos.shape[0]
for var_cat in PREDICTORES_CATEGORICOS:
  porcentajes_var_cat = (datos
    .loc[:, [var_cat]]
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
PENDIENTE RECONOCER QUÉ PREDICTORES CATEGORICOS PUEDEN CONSIDERARSE
DE TIPO ORDINAL Y CUALES NO. ESTO PARA TENER EN CLARO SI AL TRANSFORMAR
LOS DATOS PARA SU USO EN LOS MODELOS, SE EMPLEA LABEL ENCODING O ONE HOT ENCODING

CONSIDERAR CÓMO LA CLASE 'NOT IN UNIVERSE' AFECTA A ESE ORDENAMIENTO CATEGÓRICO
"""

# %%
# Porcentajes en la variable objetivo, variable categórica nominal
porcentajes_var_cat = (datos
  .loc[:, [COLUMNA_OBJETIVO]]
  .groupby([COLUMNA_OBJETIVO]).size()
  .reset_index(name = 'cantidad')
  .sort_values(by = 'cantidad', ascending = False)
  .assign(Porcentaje = lambda d: 100 * (d['cantidad'] / num_observaciones))
)

plt.figure(figsize = (12, 6))
sns.barplot(
  x = COLUMNA_OBJETIVO, y = 'Porcentaje',
  data = porcentajes_var_cat,
  order = porcentajes_var_cat[COLUMNA_OBJETIVO]
).set_title('Categoría "1" indica pago necesario de impuestos')

for index, value in enumerate(porcentajes_var_cat["Porcentaje"]):
  plt.text(index, value + 1, f'{round(value, 2)}%', ha = 'center')

plt.xticks(rotation = 45)
plt.show()

# %% [markdown]
# Note que la variable objetivo está muy desbalanceada.
# Esto debido a que la categoría que se desea predecir,
# "taxable income amount" = 1, representa aproximademente solo el
# 6% de los casos ... un porcentaje muy bajo.
# 
# Este desbalance implica que no es adecuado emplear la exactitud
# (accuracy) para concluir que un modelo predictivo es mejor que otro.
# Aquello debido, en parte, a que fácilmente se puede obtener una 
# exactitud de aproximadamente 93.79%, por medio de un modelo que 
# a cualquier observación la clasifique como que no debe pagar impuestos.
# 
# En ese sentido, requerimos alguna métrica para comparar modelos
# predictivos, apropiada para casos, como este, donde las clases de la
# variable objetivo están muy desbalanceadas.
# Para ello emplearemos el coeficiente de Kappa de Cohen.

# %% [markdown]
# ### Variables numéricas

# %%
# Número de predictores numéricos
len(PREDICTORES_NUMERICOS)

# %%
for var_num in PREDICTORES_NUMERICOS:
  fig = plt.figure(constrained_layout = True, figsize = (8, 6))
  spec = fig.add_gridspec(2, 1, height_ratios = [3, 1])

  # Histograma
  ax1 = fig.add_subplot(spec[0])
  sns.histplot(datos[var_num], kde = True, ax = ax1)
  plt.xlabel('')
  plt.ylabel('Frecuencia')
  ax1.set_title(f'Histograma y densidad estimada de variable "{var_num}"')

  # Diagrama de caja
  ax2 = fig.add_subplot(spec[1])
  sns.boxplot(x = datos[var_num], ax = ax2, orient = "h")

  iqr = (datos[[var_num]]
    .describe().T
    .assign(IQR = lambda d: d['75%'] - d['25%'])
    ['IQR'][0]
  )
  ax2.set_title(f"Rango intercuartílico: {round(iqr, 2)}")
  plt.show()

# %% [markdown]
# De los gráficos notamos que ningún predictor numérico presenta
# una distribución aproximadamente normal/gaussiana.
# 
# Asimismo, las variables "wage per hour", "capital gains",
# "capital losses" y "dividens from stocks", no solo presentan
# una gran cantidad de valores atípicos, sino que su rango
# intercuatílico es de cero. Esto significa que, para cada una de
# aquellas variables en particular, el 50% de sus observaciones
# consisten del mismo valor.
# 
# También se tiene una gran cantidad de datos atípicos para la
# variable "instance weight", pero, la forma de su distribución 
# y su cola significativa sugiere que podría resultar apropiada 
# una transformación del tipo log(1 + "instance weight"), con el fin
# de normalizar aquella variable.

# Por otro lado, se observa una mayor dispersión de los datos
# para las variables "age", "num persons worked for employer" y
# "weeks worked in year". Además, estas últimas variables no cuentan
# con valores atípicos, según el cálculo estándar vía el IQR, pues sus
# diagramas de caja no presentan puntos fuera del rango cubierto por los whiskers.

# %%
# Comparamos la homogeneidad de los predictores numéricos
(datos[PREDICTORES_NUMERICOS]
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

# %% [markdown]
# En la tabla resultante, ningún predictor numérico posee C.V menor o igual que 0.3 (30%).
# Esto significa que, para cada predictor numérico, su promedio no representa a la variable
# adecuadamente, por lo que consisten de variables heterogéneas (no homogéneas).
# 
# Esta información sería de mayor utilidad en caso se hubiese requerido imputar valores perdidos
# en los predictores numéricos, pues, hemos confirmado que una imputación vía la media aritmética
# muy probablemente no hubiese sido adecuada. Así, otro tipo de imputación se hubiese requerido.
