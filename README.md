# Calculadora Inteligente de Envíos
## Integrantes
- Ray Bolaños Aedo
- Diana Prissyla Chávez Alarcon
- Juan Diego Vizcardo Chavez
- Luciana Yangali Cáceres

## Problema asignado

Construir calculadora para predecir simultáneamente el tiempo de entrega y costo de envío para una tienda online, ayudando a clientes a tomar decisiones informadas de compra al comparar opciones de envío (express vs estándar vs económico).

## Task

- Predecir el tiempo de llegada y el costo de envío de un pedido, ambos son **valores númericos continuos.**. Nos centraremos en implementar una regresión lineal simple, por temas de complejidad utilizaremos el **modelo de Regresión Lineal Multivariable con gradiente**

- Clasificar la probabilidad de demora, en base a eso usaremos una **regresión logistica**. Ya que se clasifica de manera binaria (0 o 1).

## Metric
Dado que son dos outputs continuos, necesitamos métricas de regresión

- R^2, para los modelos de predicción
- recall, para el modelo de clasificación 

## Experience

- Para la predicción del tiempo el groudntruth será **waiting_time**
- Para la predicción del coste será el monto con que el usuario realizo la transacción
- Para la clasificación de probabilidades, se usará la columna con el mino nombre.
