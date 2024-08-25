# %%
from src.pre_procesamiento.transformar_datos_para_modelos import (
  filtrar_datos_train_test_para_modelos,
  VARS_NUM,
  VARS_CATEGORICAS
)

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix


# %%
datos_train, datos_test = filtrar_datos_train_test_para_modelos()
datos = pd.concat([datos_train, datos_test], ignore_index = True)
datos

# %%
"""
Para el modelo KNN, las variables numéricas deben estandarizarse
para que sus valores se encuentren entre 0 y 1 .

Asimismo, se transfomarán las variables categóricas vía
one-hot encoding, pero no es necesario descartar las 
columnas "extra" creadas .
"""
min_max_scaler = MinMaxScaler()
datos[VARS_NUM] = min_max_scaler.fit_transform(datos[VARS_NUM])

datos_encoded = pd.get_dummies(
  datos, 
  columns = VARS_CATEGORICAS,
  drop_first = False
)
datos_encoded

# %%
datos_train_encoded = datos_encoded.iloc[0:datos_train.shape[0]]
datos_test_encoded = datos_encoded.iloc[datos_train.shape[0]:]

datos_train_encoded

# %%
nombre_variable_target = 'class_>50K'

X_train, X_test, y_train, y_test = [
  datos_train_encoded.drop([nombre_variable_target], axis = 1).values,
  datos_test_encoded.drop([nombre_variable_target], axis = 1).values,
  datos_train_encoded[nombre_variable_target],
  datos_test_encoded[nombre_variable_target]
]

# %%
"""
Empleamos validación cruzada para estimar un candidato apropiado
del número de vecindades fijadas en el modelo KNN .
"""
cv = KFold(n_splits = 5, shuffle = True, random_state = 42)

resultados = {}
num_vecinos = [1, 3, 5, 7, 9]

for k in num_vecinos:
  knn = KNeighborsClassifier(n_neighbors = k)
  scores = cross_val_score(knn, X_train, y_train, cv = cv, scoring = 'accuracy')
  resultados[k] = {
    'exactitud_promedio': np.mean(scores),
    'desv_estandar': np.std(scores)
  }

for k, resultado in resultados.items():
  print(f'k={k}: Exactitud promedio = {resultado["exactitud_promedio"]:.4f}, Desviación estándar = {resultado["desv_estandar"]:.4f}')

# %%
# Extraer para qué valor de K se obtuvo la mayor exactitud promedio
num_optimo_de_vecinos = num_vecinos[np.argmax([
  resultado['exactitud_promedio']
  for resultado in resultados.values()
])]

num_optimo_de_vecinos

# %%
modelo = KNeighborsClassifier(n_neighbors = num_optimo_de_vecinos).fit(X_train, y_train)
print(f'Exactitud del modelo: {round(100 * modelo.score(X_test, y_test), 2)}%')

# Matríz de confusión
confusion_matrix(y_test, modelo.predict(X_test))
