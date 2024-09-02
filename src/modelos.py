# %% [markdown]
# # Modelos

# %%
from src.utils import (
  COLUMNA_OBJETIVO,
  PREDICTORES_NUMERICOS,
  PREDICTORES_CATEGORICOS
)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# %%
datos = (pd
  .read_csv('census_income/census-income.csv')
  # Excluir las columnas 'key' e 'instance weight'
  [[*PREDICTORES_NUMERICOS, *PREDICTORES_CATEGORICOS, COLUMNA_OBJETIVO]]
)

# Corregir los tipos de variables
for pred_num in PREDICTORES_NUMERICOS:
  datos[pred_num] = pd.to_numeric(datos[pred_num], errors = 'raise')

for pred_cat in PREDICTORES_CATEGORICOS:
  datos[pred_cat] = pd.Categorical(datos[pred_cat])

datos

# %% [markdown]
# ## Funciones auxiliares

# %%
# Definimos una función que calcule las métricas relevantes
# asociadas a la matriz de confusión, dado el par de parámetros
# datos_observados y datos_predichos
def mostrar_metricas_relevantes_del_modelo(y_observado, y_predicho) -> None:
  """
  Suponemos que ambos parámetros contienen solo elementos 1 y 0,
  donde 1 representa el caso exitoso. Es decir, para este proyecto,
  1 representa cuando la persona debe pagar impuesto.
  """
  tp, fn, fp, tn = confusion_matrix(y_observado, y_predicho, labels = [1, 0]).ravel()
  exactitud = (tp + tn) / (tp + fp + tn + fn)
  sensibilidad = tp / (tp + fn)
  especificidad = tn / (tn + fp)

  print('\nCantidades en la variable por predecir:', y_observado.value_counts())

  print('\nMatriz de confusión:\n', confusion_matrix(y_observado, y_predicho, labels = [1, 0]))
  print(f'\nExactitud: {exactitud}')
  print(f'\nSensibilidad: {sensibilidad}')
  print(f'\nEspecificidad: {especificidad}')

  # Coeficiente de Kappa de Cohen
  print(f'\nCoeficiente de Kappa de Cohen: {cohen_kappa_score(y_observado, y_predicho)}')

  return;

# Definimos una función para calcular la curva ROC y el AUC,
# dado un modelo, ya entrenado y capaz de predecir probabilidad
# de pertenencia a una clase; además de un cojunto de test
def graficar_curva_roc_con_auc(
  nombre_modelo, modelo, predictores_test, target_test
) -> None:
  y_pred_prob = modelo.predict_proba(predictores_test)[:, 1]

  # Calcular curva ROC y su AUC
  fpr, tpr, _ = roc_curve(target_test, y_pred_prob)
  auc_score = roc_auc_score(target_test, y_pred_prob)

  plt.figure(figsize = (8, 6))
  plt.plot(fpr, tpr, label = f"Curva ROC (AUC = {auc_score:.3f})")
  plt.plot([0, 1], [0, 1], 'k--', label = 'Clasificación aleatoria')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel("1 - Especificidad")
  plt.ylabel("Sensibilidad")
  plt.title(f"Curva ROC del modelo {nombre_modelo}")
  plt.legend(loc = "lower right")
  plt.show()

  return;

# Definimos una función para calcular el punto de corte óptimo,
# para el mejor equilibrio posible entre sensibilidad y especificidad,
# según el criterio del índice J de Youden
def optimizar_punto_de_corte_segun_criterio_de_youden(
  modelo, predictores_test, target_test
) -> float:
  y_pred_prob = modelo.predict_proba(predictores_test)[:, 1]
  fpr, tpr, thresholds = roc_curve(target_test, y_pred_prob)

  df_youden = pd.DataFrame({'thresholds' : thresholds, 'J': tpr - fpr})
  punto_de_corte_optimo = df_youden[df_youden["J"] == df_youden["J"].max()]["thresholds"].values[0]
  return punto_de_corte_optimo

# %% [markdown]
# ## Sobre la partición en entrenamiento y test
# 
# No es recomendable comparar modelos en base a un mismo conjunto
# de test. Esto porque sino podría generarse overfitting respecto
# al conjunto de test, perdiendo así generalizabilidad de los modelos.
# Aquello en el sentido que podria producirse un 'sesgo de confirmación',
# pues sería posible que, tras emplear el mismo conjunto de test para
# todos los modelos, no se obtenga el modelo que se adecúe mejor
# a otros datos nuevos, sino solo a esa muestra en particular.
# 
# En ese sentido, para cada modelo se realizará una partición
# diferente en entrenamiento y test.

# %%
# Fijamos diferentes semillas para la partición train/test
# en cada modelo
RANDOM_STATE_REG = 42
RANDOM_STATE_BOSQUE = 420
RANDOM_STATE_XGB = 6174

X = pd.get_dummies(
  datos.drop(columns = [COLUMNA_OBJETIVO]),
  columns = PREDICTORES_CATEGORICOS,
  # Descartamos una columna del total de columnas creadas por variable
  # categórica, para intentar evitar multicolinearidad entre los predictores
  drop_first = True,
  dtype = 'int64'
)

y = datos[COLUMNA_OBJETIVO]

# %% [markdown]
# ## Entrenamiento de modelos

# %% [markdown]
# ### Regresión logística

# %%
from sklearn.linear_model import LogisticRegression

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
  X, y, test_size = 0.2, random_state = RANDOM_STATE_REG
)

# %%
# Estandarizamos los predictores
estandarizador = StandardScaler()
X_train_std = estandarizador.fit_transform(X_train_reg)
X_test_std = estandarizador.transform(X_test_reg)

# %%
modelo_reg = LogisticRegression(max_iter = 1000, random_state = RANDOM_STATE_REG)
modelo_reg.fit(X_train_std, y_train_reg)

# %%
# Predicciones del modelo, con punto de corte c = 0.5
y_pred_reg = modelo_reg.predict(X_test_std)
mostrar_metricas_relevantes_del_modelo(y_test_reg, y_pred_reg)

# %%
graficar_curva_roc_con_auc(
  nombre_modelo = 'regresión logística',
  modelo = modelo_reg,
  predictores_test = X_test_std,
  target_test = y_test_reg
)

# %%
# Calcular el punto de corte para el mejor equilibrio posible
# entre sensibilidad y especificidad
punto_de_corte_optimo_reg = optimizar_punto_de_corte_segun_criterio_de_youden(
  modelo_reg, X_test_std, y_test_reg
)
print(f'Punto de corte óptimo: {punto_de_corte_optimo_reg}')

y_pred_prob_reg = modelo_reg.predict_proba(X_test_std)[:, 1]
y_pred_opt_reg = (y_pred_prob_reg >= punto_de_corte_optimo_reg) + 0

mostrar_metricas_relevantes_del_modelo(y_test_reg, y_pred_opt_reg)

# %% [markdown]
# #### Importancia de los predictores

# %%
importancia_predictores_reg = pd.DataFrame({
  'Predictor': X.columns,
  # Coeficientes estandarizados
  'Coeficiente': modelo_reg.coef_[0]
})
importancia_predictores_reg = (importancia_predictores_reg
  .assign(abs_coef = lambda d: d['Coeficiente'].abs())
  .assign(Odds_ratio = lambda d: np.exp(d['Coeficiente']))
  .sort_values(by = 'abs_coef', ascending = False)
  .loc[:, ['Predictor', 'Coeficiente', 'Odds_ratio']]
)

# Predictores, ordenados de mayor a menor importancia en la predicción
# Solo mostrar los 20 predictores más importantes
importancia_predictores_reg.head(20)

# %% [markdown]
# **Interpretación**: 
# 
# Los cinco predictores más importantes según este modelo son
# "weeks worked in year", "tax filer stat_ Nonfiler", "age", 
# "education_ Bachelors" y "sex_ Male".
# 
# Para la variable "weeks worked in year", se cumple que si
# aumenta en 1 * desviación_estándar("weeks worked in year),
# las chances que una persona deba pagar impuestos se multiplican
# por 2.524585 .
# 
# Sabemos que valores de odds_ratio mayores que 1
# indican que, cuando el predictor respectivo aumenta, 
# la probabilidad de tener que pagar impuestos se incrementa.
# 
# Por otro lado, valores de odds_ratio menores que 1
# indican que, cuando el predictor respectivo aumenta, 
# la probabilidad de tener que pagar impuestos se disminuye.
# 
# En ese sentido, como el odds_ratio de la variable binaria (1, 0)
# "tax filer stat_ Nonfiler" es menor que 1, esto implica que
# para aquellas personas cuyo valor de "tax filer stat" es "Nonfiler",
# la propabilidad de tener que pagar impuestos disminuye.
# Este comportamiento coincide con lo observado en el análisis descriptivo,
# donde se visualiza la distribución condicional de "tax filer stat",
# respecto a las clases de la variable objetivo.

# %% [markdown]
# ### Bosque aleatorio

# %%
from sklearn.ensemble import RandomForestClassifier

X_train_bosque, X_test_bosque, y_train_bosque, y_test_bosque = train_test_split(
  X, y, test_size = 0.2, random_state = RANDOM_STATE_BOSQUE
)

# %%
modelo_bosque = RandomForestClassifier(n_estimators = 100, random_state = RANDOM_STATE_BOSQUE)
modelo_bosque.fit(X_train_bosque, y_train_bosque)

# %%
y_pred_bosque = modelo_bosque.predict(X_test_bosque)
mostrar_metricas_relevantes_del_modelo(y_test_bosque, y_pred_bosque)

# %%
graficar_curva_roc_con_auc(
  nombre_modelo = 'bosque aleatorio',
  modelo = modelo_bosque,
  predictores_test = X_test_bosque,
  target_test = y_test_bosque
)

# %%
punto_de_corte_optimo_bosque = optimizar_punto_de_corte_segun_criterio_de_youden(
  modelo_bosque, X_test_bosque, y_test_bosque
)
print(f'Punto de corte óptimo: {punto_de_corte_optimo_bosque}')

y_pred_prob_bosque = modelo_bosque.predict_proba(X_test_bosque)[:, 1]
y_pred_opt_bosque = (y_pred_prob_bosque >= punto_de_corte_optimo_bosque) + 0

mostrar_metricas_relevantes_del_modelo(y_test_bosque, y_pred_opt_bosque)

# %% [markdown]
# #### Importancia de los predictores

# %%
importancia_predictores_bosque = (pd.DataFrame({
  'Predictor' : X.columns,
  'Importancia': modelo_bosque.feature_importances_
})
  .sort_values('Importancia', ascending = True)
  # Seleccionar solo los 20 predictores más importantes
  .tail(20)
)

fig, ax = plt.subplots(figsize = (6, 10))
ax.barh(
  importancia_predictores_bosque['Predictor'].values,
  importancia_predictores_bosque['Importancia'].values
)
plt.title("Importancia de los predictores en el modelo")
plt.show()

# %% [markdown]
# **Interpretación**: 
# 
# Los cinco predictores más importantes según este modelo son
# "age", "dividends from stocks", "capital gains", 
# "num  persons worked for employer" y "weeks worked in year".
# 
# Sabemos que la importancia de los predictores se calcula en
# base al **decrecimiento en impureza** que el predictor produce
# en los árboles de decisión del bosque.
# 
# Predictores con valor de importancia muy cercano
# a cero indica que no son relevantes en las predicciones.
# 
# Así, un mayor valor de importancia de un predictor indica que
# aquella variable influye más en la predicción final.

# %% [markdown]
# ### XGBoost

# %%
import re
import xgboost as xgb
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, KFold

X_xgb = X.copy()

# Remover los caracteres en los nombres de las columnas que generan
# error al entrenar el modelo
# Fuente: https://stackoverflow.com/a/50633571
regex = re.compile(r"\[|\]|<", re.IGNORECASE)
X_xgb.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in X_xgb.columns.values]

X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(
  X_xgb, y, test_size = 0.2, random_state = RANDOM_STATE_XGB
)

# %%
param_grid = { 'alpha': [0.01, 0.1, 1, 10] }

cv = KFold(n_splits = 5, shuffle = True, random_state = RANDOM_STATE_XGB)

proporcion_negativos_positivos = (y.shape[0] - y.sum()) / y.sum()

modelo_cv_xgb = xgb.XGBClassifier(
  # Calculamos la raíz cuadra de la proporción de negativos
  # y positivos, pues la cantidad de casos positivos es tan pequeña
  scale_pos_weight = np.sqrt(proporcion_negativos_positivos),
  use_label_encoder = False,
  random_state = RANDOM_STATE_XGB
)

kappa_scorer = make_scorer(cohen_kappa_score)

grid_search = GridSearchCV(
  estimator = modelo_cv_xgb, 
  param_grid = param_grid, 
  cv = cv, 
  scoring = kappa_scorer, 
  n_jobs = -1,
  verbose = 1
)
grid_search.fit(X_train_xgb, y_train_xgb)

# %%
modelo_xgb = grid_search.best_estimator_
modelo_xgb.fit(X_train_xgb, y_train_xgb)

# %%
y_pred_xgb = modelo_xgb.predict(X_test_xgb)
mostrar_metricas_relevantes_del_modelo(y_test_xgb, y_pred_xgb)

# %%
graficar_curva_roc_con_auc(
  nombre_modelo = 'XGB',
  modelo = modelo_xgb,
  predictores_test = X_test_xgb,
  target_test = y_test_xgb
)

# %%
punto_de_corte_optimo_xgb = optimizar_punto_de_corte_segun_criterio_de_youden(
  modelo_xgb, X_test_xgb, y_test_xgb
)
print(f'Punto de corte óptimo: {punto_de_corte_optimo_xgb}')

y_pred_prob_xgb = modelo_xgb.predict_proba(X_test_xgb)[:, 1]
y_pred_opt_xgb = (y_pred_prob_xgb >= punto_de_corte_optimo_xgb) + 0

mostrar_metricas_relevantes_del_modelo(y_test_xgb, y_pred_opt_xgb)

# %% [markdown]
# #### Importancia de los predictores

# %%
importancia_predictores_xgb = (pd.DataFrame({
  # No es necesario extraer las columns de X_xgb, pues tienen
  # el mismo orden que en X. Emplearemos las columnas de X 
  # por consistencia con los modelos previos.
  'Predictor' : X.columns,
  'Importancia': modelo_xgb.feature_importances_
})
  .sort_values('Importancia', ascending = True)
  # Seleccionar solo los 20 predictores más importantes
  .tail(20)
)

fig, ax = plt.subplots(figsize = (6, 10))
ax.barh(
  importancia_predictores_xgb['Predictor'].values,
  importancia_predictores_xgb['Importancia'].values
)
plt.title("Importancia de los predictores en el modelo")
plt.show()

# %% [markdown]
# **Interpretación**: 
# 
# Los cinco predictores más importantes según este modelo son
# "tax filer stat_ Nonfiler", "weeks worked in year", "sex_ Male",
# "major occupation code_ Professional speciality" y 
# "major occupation code_ Other service".
# 
# La importancia de los predictores se calcula diferente al modelo
# bosque aleatorio. Para el modelo XGBoost, los predictores con mayor
# importancia son aquellos que más **incrementan la exactitud** del modelo
# cuando se emplea en los árboles de decisión que componen el modelo.

# Aún así, un mayor valor de importancia de un predictor indica que
# aquella variable influye más en la predicción final.

# %% [markdown]
# ## Guardamos los modelos

# %%
import pickle

# %%
with open("modelos/modelo_reg.pkl", "wb") as model_file:
  pickle.dump(modelo_reg, model_file)

with open("modelos/modelo_bosque.pkl", "wb") as model_file:
  pickle.dump(modelo_bosque, model_file)

with open("modelos/modelo_xgb.pkl", "wb") as model_file:
  pickle.dump(modelo_xgb, model_file)

# %% [markdown]
# ## Comparación de los modelos

# %%
with open("modelos/modelo_reg.pkl", "rb") as model_file:
  modelo_reg = pickle.load(model_file)

with open("modelos/modelo_bosque.pkl", "rb") as model_file:
  modelo_bosque = pickle.load(model_file)

with open("modelos/modelo_xgb.pkl", "rb") as model_file:
  modelo_xgb = pickle.load(model_file)

# %% [markdown]
# ### Curvas ROC y valor AUC

# %%
from sklearn.metrics import auc

plt.figure(figsize = (10, 8))

modelos = [
  {
    'nombre' : 'Regresión logística',
    'modelo' : modelo_reg,
    'X_test' : X_test_std,
    'y_test' : y_test_reg
  },
  {
    'nombre' : 'Bosque aleatorio',
    'modelo' : modelo_bosque,
    'X_test' : X_test_bosque,
    'y_test' : y_test_bosque
  },
  {
    'nombre' : 'XGBoost',
    'modelo' : modelo_xgb,
    'X_test' : X_test_xgb,
    'y_test' : y_test_xgb
  }
]

for modelo_info in modelos:
  y_prob_pred = modelo_info['modelo'].predict_proba(modelo_info['X_test'])[:, 1]
  
  fpr, tpr, _ = roc_curve(modelo_info['y_test'], y_prob_pred)
  roc_auc = auc(fpr, tpr)
  
  plt.plot(
    fpr, tpr, label = f'{modelo_info["nombre"]} (AUC = {roc_auc:.3f})'
  )

plt.plot([0, 1], [0, 1], 'k--', label = 'Clasificación aleatoria')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("1 - Especificidad")
plt.ylabel("Sensibilidad")
plt.title('Curvas ROC de los modelos entrenados')
plt.legend(loc = "lower right")
plt.grid()
plt.show()

# %% [markdown]
# Del gráfico, notamos que las curvas ROC más cercanas al punto (0, 1),
# que representa una exactitud total de 100%, son la asociada a XGBoost;
# luego, la curva respectiva al bosque aleatorio; y, por último, la curva
# ROC del modelo de regresión logística.
# 
# Esto sugiere que el modelo XGBoost podría ser más apropiado para la
# predicción.
# 
# Por otro lado, los AUC de los modelos son altos, pues son mayores
# que 0.9. Asimismo, el máximo AUC se halló para el modelo XGBoost, lo
# cual refuerza su elección como el modelo por seleccionar.

# %% [markdown]
# ### Especificidad y el coeficiente de Kappa

# %%
metricas_modelos = pd.DataFrame({
  'Modelo' : [modelo_info['nombre'] for modelo_info in modelos],
  'Exactitud': [None, None, None],
  'Sensibilidad': [None, None, None],
  'Especificidad': [None, None, None],
  'Coef_Kappa_Cohen': [None, None, None]
})

for modelo_index in metricas_modelos.index:
  y_observado = modelos[modelo_index]['y_test']
  y_predicho = modelos[modelo_index]['modelo'].predict(modelos[modelo_index]['X_test'])

  tp, fn, fp, tn = confusion_matrix(y_observado, y_predicho, labels = [1, 0]).ravel()

  metricas_modelos.loc[modelo_index, 'Exactitud'] = (tp + tn) / (tp + fp + tn + fn)
  metricas_modelos.loc[modelo_index, 'Sensibilidad'] = tp / (tp + fn)
  metricas_modelos.loc[modelo_index, 'Especificidad'] = tn / (tn + fp)
  metricas_modelos.loc[modelo_index, 'Coef_Kappa_Cohen'] = cohen_kappa_score(y_observado, y_predicho)

metricas_modelos

# %% [markdown]
# Entre los tres modelos, XGBoost produjo el mayor coeficiente de Kappa de 
# Cohen, con un valor de aproximadamente 0.59, lo cual indica un nivel
# **moderado** de acuerdo entre las observaciones y predicciones.
# 
# En ese sentido, el modelo XGBoost nos provee predicciones
# que no se deben simplemente al azar ... hecho que se advirtió
# en el análisis descriptivo, debido a que el porcentaje de personas
# que sí deben pagar impuestos es tan pequeño (aproximadamente 6%).
# 
# Asimismo, resaltamos el hecho que el modelo XGBoost balancea mejor
# que los otros dos modelos la sensibilidad y especificidad, produciendo
# incluso valores mayores que 70% para ambas métricas.
# 
# Por último, destacamos el hecho que, pese al balance entre sensibilidad
# y especificidad del modelo XGBoost, este produce una especificidad 
# relativamente alta (aproximadamente 96%). Esto significa que el 
# modelo XGBoost solo asigna erróneamente a 4% de las personas que no
# tienen que pagar impuestos, como si sí tuviesen que pagar impuestos.
# 
# Minimizar aquel escenario (error de tipo I) es de mayor prioridad que
# minimizar los errores de tipo II (asignar como que no debe pagar impuesto
# a alquien que sí debe pagar), en el **contexto particular** del conjunto de datos.
# 
# Aquello por la posibilidad de demanda legal por parte de personas que
# se hubiesen visto en la obligación de tener que pagar impuesto,
# según la predicción del modelo, pese a que no era su responsabilidad.

# %% [markdown]
# ## Selección del modelo
# 
# En base a lo expuesto previamente, seleccionamos el modelo XGBoost
# como el más adecuado para la predicción de si una persona debe 
# pagar impuestos.
