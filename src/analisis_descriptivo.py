# %% [markdown]
# # Análisis descriptivo

# %%
from src.utils import (
  COLUMNA_OBJETIVO,
  VARIABLES_NUMERICAS,
  VARIABLES_CATEGORICAS,
  PREDICTORES_CATEGORICOS
)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# %%
datos = (pd
  .read_csv('census_income/census-income.csv')
  # Descartar la columna de identificador de filas (primary key)
  [[*VARIABLES_NUMERICAS, *PREDICTORES_CATEGORICOS, COLUMNA_OBJETIVO]]
)

# Corregir los tipos de variables
for var_numerica in VARIABLES_NUMERICAS:
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

# %% [markdown]
# A partir de l


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
# Número de variables numéricas
len(VARIABLES_NUMERICAS)

# %%
for var_num in VARIABLES_NUMERICAS:
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
# De los gráficos notamos que ninguna variable numérica presenta
# una distribución aproximadamente normal/gaussiana.
# 
# Asimismo, las variables "wage per hour", "capital gains",
# "capital losses" y "dividens from stocks", no solo presentan
# una gran cantidad de valores atípicos, sino que su rango
# intercuatílico es cero. Esto significa que, para cada una de
# aquellas variables en particular, el 50% de sus observaciones
# consisten del mismo valor.
# 
# También se tiene una gran cantidad de datos atípicos para la
# variable "instance weight", pero, la forma de su distribución 
# y su cola significativa sugiere que podría resultar apropiada 
# una transformación del tipo log(1 + "instance weight"), con el fin
# de normalizar aquella variable. Esto en caso fuese a emplearse
# tal variable para el entrenamiento de los modelos, mas, en el 
# archivo `census_income/census-income_doc.txt` se sugiere que no
# se utilice aquella variable para los modelos de clasificación.

# Por otro lado, se observa una mayor dispersión de los datos
# para las variables "age", "num persons worked for employer" y
# "weeks worked in year". Además, estas últimas variables no cuentan
# con valores atípicos, según el cálculo estándar vía el IQR, pues sus
# diagramas de caja no presentan puntos fuera del rango cubierto por los whiskers.

# %%
# Comparamos la homogeneidad de los variables numéricas
(datos[VARIABLES_NUMERICAS]
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
# En la tabla resultante, ninguna varable numérica posee C.V menor o igual que 0.3 (30%).
# Esto significa que, para cada variable numérica, su promedio no representa a la variable
# adecuadamente, por lo que consisten de variables heterogéneas (no homogéneas).
# 
# Esta información sería de mayor utilidad en caso se hubiese requerido imputar valores perdidos
# en las variables numéricas, pues, hemos confirmado que una imputación vía la media aritmética
# muy probablemente no hubiese sido adecuada. Así, otro tipo de imputación se hubiese requerido.

# %% [markdown]
# ## Análisis entre variables

# %% [markdown]
# ### Comparación entre predictores categóricos y variable objetivo

# %%
for pred_cat in PREDICTORES_CATEGORICOS:
  frecuencias = datos.groupby([COLUMNA_OBJETIVO, pred_cat]).size().reset_index(name = 'Cantidad')

  # Frecuencias de la variable objetivo en cada clase del predictor categórico
  plt.figure(figsize = (12, 6))
  sns.barplot(x = COLUMNA_OBJETIVO, y = 'Cantidad', hue = pred_cat, data = frecuencias)
  plt.title(f'Frecuencias de variable objectivo, por {pred_cat}')
  plt.xticks(rotation = 45)
  plt.show()

  # Frecuencias de las clases del predictor categórico, en cada clase de la variable objetivo
  plt.figure(figsize = (12, 6))
  sns.barplot(x = pred_cat, y = 'Cantidad', hue = COLUMNA_OBJETIVO, data = frecuencias)
  plt.title(f'Frecuencias de {pred_cat}, por {COLUMNA_OBJETIVO}')
  plt.xticks(rotation = 45)
  plt.show()

# %% [markdown]
# ### Comparación entre variables numéricas y variable objetivo

# %%
for var_num in VARIABLES_NUMERICAS:
  plt.figure(figsize = (12, 6))
  sns.violinplot(x = COLUMNA_OBJETIVO, y = var_num, data = datos, inner = None)
  sns.boxplot(
    x = COLUMNA_OBJETIVO, 
    y = var_num, 
    data = datos, 
    width = 0.25,
    fill = False
  )
  plt.title(f'Distribución de "{var_num}", según valor de la variable objetivo')
  plt.show()

# %%
# Correlación entre las variables numéricas
matriz_de_correlaciones = datos[VARIABLES_NUMERICAS].corr()

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
sns.pairplot(datos[VARIABLES_NUMERICAS])

# %%
# Análisis de multicolinearidad entre predictores numéricos
X = add_constant(datos[VARIABLES_NUMERICAS])

vif_data = pd.DataFrame()
vif_data["predictor"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# No se requiere la variable artificial "const" como predictor
vif_data.iloc[1:]
