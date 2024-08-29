# %%
from src.pre_procesamiento.transformar_datos_para_modelos import (
  filtrar_datos_train_test_para_modelos,
  VARS_CATEGORICAS
)

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, cohen_kappa_score


# %%
datos_train, datos_test = filtrar_datos_train_test_para_modelos()
datos = pd.concat([datos_train, datos_test], ignore_index = True)
datos

# %%
# Transformar las variables categóricas via variables dummy/binarias
datos_encoded = pd.get_dummies(
  datos, 
  columns = VARS_CATEGORICAS,
  # Descartamos una columna del total de columnas creadas por variable
  # categórica, para intentar evitar multicolinearidad entre los predictores
  drop_first = True,
  dtype = 'int64'
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
modelo = LogisticRegression().fit(X_train, y_train)
print(f'Exactitud del modelo: {round(100 * modelo.score(X_test, y_test), 2)}%')

# %%
"""
Comparemos la exactitud del modelo si transformamos los 
predictores numéricos via Z-score
"""
estandarizador = StandardScaler()
X_train_std = estandarizador.fit_transform(X_train)
X_test_std = estandarizador.transform(X_test)

modelo = LogisticRegression().fit(X_train_std, y_train)
print(f'Exactitud del modelo tras estandarización vía Z-score: {round(100 * modelo.score(X_test_std, y_test), 2)}%')

print('\nCantidades en la variable por predecir:')
print(y_test.value_counts())

# Predicciones del modelo
y_pred = modelo.predict(X_test_std)

# Matríz de confusión
confusion_matrix(y_test, y_pred, labels = [1, 0])

# %%
# Coeficiente de Kappa de Cohen
cohen_kappa_score(y_test, y_pred)

# %%
"""
Comparar qué predictores tienen mayor efecto sobre 
la variable objetivo, 'class_>50K'
"""
importance_df_scaled = pd.DataFrame({
  'Predictor': datos_train_encoded.columns.difference([nombre_variable_target]),
  # Coeficientes estandarizados
  'Coeficiente': modelo.coef_[0]
})
importance_df_scaled = (importance_df_scaled
  .assign(abs_coef = lambda d: d['Coeficiente'].abs())
  .sort_values(by = 'abs_coef', ascending = False)
  .loc[:, ['Predictor', 'Coeficiente']]
)

importance_df_scaled.head(10)
