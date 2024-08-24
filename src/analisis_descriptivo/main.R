#### Cargar datos ####
datos_train <- read.csv(
  'data/datos_train_limpios_con_vacios.csv', sep = ',', encoding = 'UTF-8', check.names = FALSE
)
head(datos_train)
str(datos_train)

datos_test <- read.csv(
  'data/datos_test_limpios_con_vacios.csv', sep = ',', encoding = 'UTF-8', check.names = FALSE

)
head(datos_test)

# Reconstruir dataset original
datos <- rbind(datos_train, datos_test)
str(datos)

#### Asignar tipo de dato factor a las variables categóricas ####
vars_cat <- c(
  'workclass',         
  'education',         
  'marital-status',
  'occupation',        
  'relationship',      
  'race',              
  'sex',               
  'native-country',
  'class'
)
for (var_cat in vars_cat) {
  # Por variable categórica, considerar el grupo de 
  # observaciones vacías como parte de una misma categoría
  datos[, var_cat] <- addNA(datos[, var_cat], ifany = TRUE)
}
str(datos)


#### Comparar la variable por predecir, class, con el resto de variables, uno a uno ####
library(ggplot2)

## Comparación entre variables categóricas ##

# Fijar alguna variable categórica distinta de 'class'
var_cat <- 'workclass'         
# var_cat <- 'education'         
# var_cat <- 'marital-status'
# var_cat <- 'occupation'        
# var_cat <- 'relationship'      
# var_cat <- 'race'              
# var_cat <- 'sex'               
# var_cat <- 'native-country'

proporciones <- data.frame(table(datos[, 'class'], datos[, var_cat]))
names(proporciones) <- c("class", var_cat, "Cantidad")

# Proporciones de 'class' en cada valor de la variable categórica seleccionada
ggplot(proporciones) + 
  aes(x = class, y = Cantidad, fill = .data[[var_cat]]) +
  geom_bar(stat = "identity")

# Proporciones de la variable categórica seleccionada en cada valor de 'class'
ggplot(proporciones) + 
  aes(x = .data[[var_cat]], y = Cantidad, fill = class) +
  geom_bar(stat = "identity")


## Comparación con variables numéricas ##

# Fijar alguna variable numérica
var_num <- "age"
# var_num <- "fnlwgt"
# var_num <- "education-num"
# var_num <- "capital-gain"
# var_num <- "capital-loss"
# var_num <- "hours-per-week"

# Gráfico de violin de la variable numérica, por categoría de 'class'
ggplot(datos) +
  aes(x = class, y = .data[[var_num]], group = class) +
  geom_violin(aes(fill = class), width = 1) +
  geom_boxplot(alpha = 0.5, width = 0.1) + 
  theme(legend.position = "top")
