# Proyecto Timi

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

- Los nombres de las columnas están especificadas en `census+income/old.adult.names` .

- Filas/observaciones: 
  - Entrenamiento: `census+income/adult.data`
  - Test: `census+income/adult.test`


## Reproducibilidad

- Se empleó Python 3.11.
- Los paquetes empleados están especificadas en `requirements.txt` .
