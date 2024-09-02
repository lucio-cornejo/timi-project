# %% [markdown]
# # Modelos

# %%
from src.utils import (
  COLUMNA_OBJETIVO,
  PREDICTORES_NUMERICOS,
  PREDICTORES_CATEGORICOS
)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# %%
datos = (pd
  .read_csv('census_income/census-income.csv')
  # Excluir las columnas 'key' e 'instance weight'
  [[*PREDICTORES_NUMERICOS, *PREDICTORES_CATEGORICOS, COLUMNA_OBJETIVO]]
)
# ).iloc[0:1000]

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
  fpr, tpr, thresholds = roc_curve(target_test, y_pred_prob)
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
RANDOM_STATE_MODELO_3 = 6174

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
# ## Regresión logística

# %%
from sklearn.linear_model import LogisticRegression

# %%
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
  X, y, test_size = 0.2, random_state = RANDOM_STATE_REG
)

# %%
# Estandarizamos los predictores
estandarizador = StandardScaler()
X_train_std = estandarizador.fit_transform(X_train_reg)
X_test_std = estandarizador.transform(X_test_reg)

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
# ### Importancia de los predictores

# %%
importance_df_scaled = pd.DataFrame({
  'Predictor': X.columns,
  # Coeficientes estandarizados
  'Coeficiente': modelo_reg.coef_[0]
})
importance_df_scaled = (importance_df_scaled
  .assign(abs_coef = lambda d: d['Coeficiente'].abs())
  .sort_values(by = 'abs_coef', ascending = False)
  .loc[:, ['Predictor', 'Coeficiente']]
)

importance_df_scaled.head(20)

# %% [markdown]
# **Interpretación**: 
"""
NO OLVIDAR QUE LOS BETA_I SE INTERPRETAN EN BASE Al CAMBIO DEL 
PREDICTOR X_I EN DESVIACION_ESTANDAR(X_I), NO CAMBIO EN 1 NOMÁS.
"""

# %% [markdown]
# ## Bosque aleatorio

# %%
from sklearn.ensemble import RandomForestClassifier

# %%
X_train_bosque, X_test_bosque, y_train_bosque, y_test_bosque = train_test_split(
  X, y, test_size = 0.2, random_state = RANDOM_STATE_BOSQUE
)

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
# ### Importancia de los predictores

# %%
importancia_predictores = (pd.DataFrame({
  'Predictor' : X.columns,
  'Importancia': modelo_bosque.feature_importances_
})
  .sort_values('Importancia', ascending = True)
  # Seleccionar solo los 20 predictores más importantes
  .tail(20)
)

fig, ax = plt.subplots(figsize = (6, 10))
ax.barh(
  importancia_predictores['Predictor'].values,
  importancia_predictores['Importancia'].values
)
plt.title("Importancia de los predictores en el modelo")
plt.show()
