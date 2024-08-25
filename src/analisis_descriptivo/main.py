# %%
import pandas as pd

# Load the training and test datasets
datos_train = pd.read_csv('data/datos_train_limpios_con_vacios.csv', encoding='utf-8')
datos_test = pd.read_csv('data/datos_test_limpios_con_vacios.csv', encoding='utf-8')

# View the first few rows and the structure of the data
print(datos_train.head())
print(datos_train.info())

print(datos_test.head())
print(datos_test.info())

# Combine the datasets
datos = pd.concat([datos_train, datos_test], ignore_index=True)
print(datos.info())

# %%
# List of categorical variables
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

# Convert columns to categorical and handle missing values
for var_cat in vars_cat:
    datos[var_cat] = pd.Categorical(datos[var_cat])

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Set up a categorical variable for comparison
var_cat = 'workclass'  
# var_cat = 'education'         
# var_cat = 'marital-status'
# var_cat = 'occupation'        
# var_cat = 'relationship'      
# var_cat = 'race'              
# var_cat = 'sex'               
# var_cat = 'native-country'

# Create a DataFrame with counts
proporciones = datos.groupby(['class', var_cat]).size().reset_index(name='Cantidad')

# Proportions of 'class' within each category of the chosen variable
plt.figure(figsize=(12, 6))
sns.barplot(x='class', y='Cantidad', hue=var_cat, data=proporciones)
plt.title(f'Proportions of class by {var_cat}')
plt.show()

# Proportions of the chosen variable within each class
plt.figure(figsize=(12, 6))
sns.barplot(x=var_cat, y='Cantidad', hue='class', data=proporciones)
plt.title(f'Proportions of {var_cat} by class')
plt.show()

# %%
# Set up a numerical variable for comparison
var_num = 'age'
# var_num = "fnlwgt"
# var_num = "education-num"
# var_num = "capital-gain"
# var_num = "capital-loss"
# var_num = "hours-per-week"

# Create a violin plot of the numerical variable by 'class'
plt.figure(figsize=(12, 6))
sns.violinplot(x='class', y=var_num, data=datos, inner=None, palette='muted')
sns.boxplot(x='class', y=var_num, data=datos, width=0.1, palette='dark')
plt.title(f'Distribution of {var_num} by class')
plt.show()
