## Análisis crítico y explicación de resultados

¿El modelo mantiene un rendimiento consistente?

El modelo mantiene un rendimiento consistente de enero del 2020 hasta marzo de 2020. luego en el mes de abril en adelante, cae su F1.

¿Qué factores podrían explicar la variación en el desempeño?

Factores que influyen en su desempeño y en base a la fecha de la data se relaciona directamente con el inicio de la Pandemia de >COVI>D-19 en la ciudad de Nueva York, lo que afecta los patrones de viaje en taxi y el escenario cambia totalmente. Cambios enel comportamiento de los usuarios, distancia y duración de los viajes, cambios en las propinas y reducción de la cantidad de viajes. Lo anterior, impacta de manera directa en el rendimiento del modelo. 

¿Qué acciones recomendarías para mejorar la robustez del modelo en el tiempo?

Reentrenamiento incremental del modelo para ir actualizando con datos más recientes, así el modelo se irá "adaptando" a la nueva realidad. También se podría realizar feature engineering para crear caraterísticas que puedan capturar cambios estacionales, tendencias. Configurar alertas cuando el F1 caiga por debajo de algun umbral predefinido. 
