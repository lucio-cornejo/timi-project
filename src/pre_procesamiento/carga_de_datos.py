import pandas as pd


def cargar_datos_train_test() -> tuple[pd.DataFrame]:
  """
  Carga de datos, con los nombres de las columnas

  @return: (datos de entrenamiento, datos de test)
  """
  datos_train = pd.read_table(
    'census+income/adult.data',
    header = None,
    delimiter = ',',
    skip_blank_lines = True,
    # Evitar leer '?' como ' ?'
    skipinitialspace = True
  )

  datos_test = pd.read_table(
    'census+income/adult.test',
    header = None,
    delimiter = ',',
    skip_blank_lines = True,
    skipinitialspace = True,
    # Ignorar línea con valor '|1x3 Cross validator'
    skiprows = 1
  )

  # Especificamos los nombres de columnas, como 
  # se describe en el archivo 'census+income/old.adult.names'
  NOMBRES_COLUMNAS = [
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education-num',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
    'native-country',
    'class'
  ]

  datos_train.columns = NOMBRES_COLUMNAS
  datos_test.columns = NOMBRES_COLUMNAS

  return datos_train, datos_test
