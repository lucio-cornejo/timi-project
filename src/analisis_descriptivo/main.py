# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %%
# Cargar datos de entrenamiento y de test
datos_train = pd.read_csv('data/datos_train_limpios_con_vacios.csv', encoding='utf-8')
datos_test = pd.read_csv('data/datos_test_limpios_con_vacios.csv', encoding='utf-8')

# %%
datos_train.info()

# %%
datos_test.info()

# %%
# Reconstruir el conjunto de datos
datos = pd.concat([datos_train, datos_test], ignore_index=True)
datos.info()

# %%
# Variables categóricas
vars_cat = [
  'workclass',         
  'education',         
  'marital-status',
  'occupation',        
  'relationship',      
  'race',              
  'sex',               
  'native-country',
  'class'
]

for var_cat in vars_cat:
  datos[var_cat] = pd.Categorical(datos[var_cat])

# %%
"""
Comparar variables categóricas con la variable por predecir
"""
for var_cat in vars_cat:
  if var_cat == 'class': continue

  frecuencias = datos.groupby(['class', var_cat]).size().reset_index(name='Cantidad')

  # Frecuencias de 'class' en cada variable categórica diferente
  plt.figure(figsize = (12, 6))
  sns.barplot(x = 'class', y = 'Cantidad', hue = var_cat, data = frecuencias)
  plt.title(f'Frecuencias de "class", por {var_cat}')
  plt.show()

  # Frecuencias de las otras variables categóricas, en cada categoría de 'class'
  plt.figure(figsize = (12, 6))
  sns.barplot(x = var_cat, y = 'Cantidad', hue = 'class', data = frecuencias)
  plt.title(f'Frecuencias de {var_cat} por "class"')
  plt.show()

# %%
"""
Comparar variables numéricas con la variable por predecir
"""
# Variables numéricas
vars_num = [
  'age',
  "fnlwgt",
  "education-num",
  "capital-gain",
  "capital-loss",
  "hours-per-week"
] 
for var_num in vars_num:
  plt.figure(figsize = (12, 6))
  sns.violinplot(x = 'class', y = var_num, data = datos, inner = None, palette = 'muted')
  sns.boxplot(x = 'class', y = var_num, data = datos, width = 0.1)
  plt.title(f'Distribution of {var_num} by class')
  plt.show()
