# Proyecto Timi

## Notebooks

1. `src/notebooks/pre_procesamiento.ipynb`
1. `src/notebooks/analisis_descriptivo.ipynb`

## Recomendaciones del informe

- En vez de usar la exactitud, para comparar modelos, emplear una métrica
que considere el caso en que las clases de la variable por predecir están
desbalanceadas. Una métrica que puede servir ese fin es el coeficiente de Kappa de Cohen.

Explicación:

Considera un modelo de detección de fraude en el que sólo el 1% de las transacciones sean fraudulentas (clase minoritaria). Un modelo que siempre predice "no fraude" (la clase mayoritaria) tendría una precisión del 99% pero fallaría por completo en la detección del fraude, lo que haría que el modelo fuera ineficaz para el propósito previsto.


- Comparar sus hipótesis con lo descubierto vía los modelos.

- Página 15
    - Quitar secciones 
      `Relación entre edad y horas trabajadas`,
      `Correlación entre educación y ganancias de capital`,
      `Relación entre peso final de la muestra (fnlwgt) y otras variables`,
      `Horas trabajadas por semana y ganancias de capital`,
      pues la correlación es muy pequeña como para poder inferir algo.

    - Mencionar las colas significativas, sth like:
      Las variables fnlwgt, capital-gain y capital-loss presentan
      distribuciones con colas muy significativas. Esto sugiere
      que podría ser necesario transformar tales variables,
      por ejemplo calculando logaritmo, antes de emplearlas en los modelos.

- Página 17
    - `Distribución general`
        - Pusieron que education-num es más disperso para `<=50k`,
        pero es para `>50K`, pues el rango intercuartílico es mayor.
    - `Mediana y Rango Intercuartílico (IQR)`
        - Mencionarles que el rango intercuartilico es un numero,
        no un intervalo/rango.
    - `Conclusiones`
        - Mencionarles tener cuidado de usar el término `correlación`
        cuando no nos referimos a una relación lineal.

- Página 19
    - Mencionan que se hizo retención de sólo de las variables más relevantes
    para el modelo; pero, en el código no se llegó a implementar aquello.

- Página 20
    - 


## General

- Las matrices de confusión se han creado siguiendo el siguiente
orden entre filas y columnas: 

![](https://cdn.prod.website-files.com/660ef16a9e0687d9cc27474a/662c42677529a0f4e97e4f96_644aea65cefe35380f198a5a_class_guide_cm08.png)

## Algunas posibles mejoras

### Limpieza de datos

- Tratamiento de los valores perdidos
    - Por ejemplo: Reemplazar valor perdido con categoría más frecuente en tal variable.

- Tratamiendo de los valores atípicos

- ¿Tomar logaritmo a las variables numéricas con distribuicones de colas muy pronunciadas?

### Análisis descriptivo

- Plantear hipótesis de qué variables podrían servir para predecir si se debe pagar impuestos.

- Hacer los mismos gráficos del archivo `analisis_descriptivo/main.ipynb`, tanto para
los datos de entrenamiento como los datos de test. Esto pues las distribuciones deberían
ser similares.

### Modelos

- Usar AUC para mejorar a exactitud del modelo de bosque aleatorio.
- Por cada modelo, seleccionar las variables que más influyeron y reentrenar el modelo
para notar el cambio en la exactitud.

- Testear KNN, con parametro 3 =numero de vecinos, pero para diferentes datasets de test,
no solo el especificado en el ZIP .

## Base de datos

Archivo `census_income/census-income.csv`.

## Reproducibilidad

- Se empleó Python 3.11.
- Los paquetes empleados están especificadas en `requirements.txt` .
