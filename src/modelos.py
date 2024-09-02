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
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score
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
  datos[pred_num] = pd.to_numeric(datos[pred_num], errors = 'coerce')

for pred_cat in PREDICTORES_CATEGORICOS:
  datos[pred_cat] = pd.Categorical(datos[pred_cat])

datos

# %%
# Definimos una función que calcule las métricas relevantes
# asociadas a la matriz de confusión, dado el par de parámetros
# datos_observados y datos_predichos
def mostrar_metricas_relevantes_del_modelo(y_observado, y_predicho) -> None:
  """
  Se supone que ambos parámetros contienen solo elementos 1 y 0,
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

# %% [markdown]
# ## Regresión logística

# %%
from sklearn.linear_model import LogisticRegression

# %%
X = pd.get_dummies(
  datos.drop(columns = [COLUMNA_OBJETIVO]),
  columns = PREDICTORES_CATEGORICOS,
  # Descartamos una columna del total de columnas creadas por variable
  # categórica, para intentar evitar multicolinearidad entre los predictores
  drop_first = True,
  dtype = 'int64'
)

y = datos[COLUMNA_OBJETIVO]

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# %%
# Estandarizamos los predictores
estandarizador = StandardScaler()
X_train_std = estandarizador.fit_transform(X_train)
X_test_std = estandarizador.transform(X_test)

modelo_reg = LogisticRegression(max_iter = 1000, random_state = 42)
modelo_reg.fit(X_train_std, y_train)

# %%
# Predicciones del modelo, con punto de corte c = 0.5
y_pred = modelo_reg.predict(X_test_std)

mostrar_metricas_relevantes_del_modelo(y_test, y_pred)

# %%
# Probabilidades de pagar impuesto
y_pred_prob = modelo_reg.predict_proba(X_test_std)[:, 1]

# Calcular curva ROC y su AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
auc_score = roc_auc_score(y_test, y_pred_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label = f"Curva ROC (AUC = {auc_score:.3f})")
plt.plot([0, 1], [0, 1], 'k--', label = 'Clasificación aleatoria')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("1 - Especificidad")
plt.ylabel("Sensibilidad")
plt.title("Curva ROC para regresión logística")
plt.legend(loc = "lower right")
plt.show()

# %%
# Calcular el punto de corte para el mejor equilibrio posible
# entre sensibilidad y especificidad
df_youden = pd.DataFrame({'thresholds' : thresholds, 'J': tpr - fpr})
punto_de_corte_optimo = df_youden[df_youden["J"] == df_youden["J"].max()]["thresholds"].values[0]
punto_de_corte_optimo

# %%
y_pred_opt = (y_pred_prob >= punto_de_corte_optimo) + 0

mostrar_metricas_relevantes_del_modelo(y_test, y_pred_opt)

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
# ## Bosque aleatorio
