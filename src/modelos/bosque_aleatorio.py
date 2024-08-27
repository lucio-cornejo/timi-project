# %%
from src.pre_procesamiento.transformar_datos_para_modelos import (
  filtrar_datos_train_test_para_modelos,
  VARS_CATEGORICAS
)

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

# %%
datos_train, datos_test = filtrar_datos_train_test_para_modelos()
datos = pd.concat([datos_train, datos_test], ignore_index = True)
datos

# %%
"""
Para el modelo de bosque aleatorio, no es necesario realizar
one-hot encoding de las variables categóricas.

Sin embargo, por cuestiones de la librería usada para trabajar con
tal modelo, se requiere convertir en valores numéricos a los 
valores de cada variable categórica.
"""
le = LabelEncoder()
datos_encoded = datos.copy()
for var_cat in VARS_CATEGORICAS:
  datos_encoded[var_cat] = le.fit_transform(datos[var_cat])

datos_encoded

# %%
datos_train_encoded = datos_encoded.iloc[0:datos_train.shape[0]]
datos_test_encoded = datos_encoded.iloc[datos_train.shape[0]:]

datos_train_encoded

# %%
nombre_variable_target = 'class'

X_train, X_test, y_train, y_test = [
  datos_train_encoded.drop([nombre_variable_target], axis = 1).values,
  datos_test_encoded.drop([nombre_variable_target], axis = 1).values,
  datos_train_encoded[nombre_variable_target],
  datos_test_encoded[nombre_variable_target]
]

# %%
modelo = RandomForestClassifier(n_estimators = 100).fit(X_train, y_train)
print(f'Exactitud del modelo: {round(100 * modelo.score(X_test, y_test), 2)}%')

print('\nCantidades en la variable por predecir:')
print(y_test.value_counts())

# Matríz de confusión
confusion_matrix(y_test, modelo.predict(X_test), labels = [1, 0])

# %%
"""
Comparar qué predictores tienen mayor efecto sobre la variable objetivo, 'class'
"""
sorted_idx = modelo.feature_importances_.argsort()

plt.barh(
  datos_train_encoded.columns.difference([nombre_variable_target])[sorted_idx],
  modelo.feature_importances_[sorted_idx]
)
plt.xlabel("Importancia de los predictores en el modelo")
