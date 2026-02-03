# Simule-a-los-piglins-en-python
Este código viene del video de los piglins que subi a mi canal: https://youtu.be/X0a6A_n5nJg

El proyecto se compone de 6 scripts y 2 .CSV
Los dos CSV son producto de simulaciones, resultados como producto de la 1.19 y resultadosV16 de la 1.16.1, si no quieres correr las simulaciones porque es demasiado tiempo los he dejado igualmente.

## Proceso de ejecución
### 1. Correr Simulador.py
En este script simulamos el tradeo de los piglins y guardamos en el csv cuántos tradeos tomó conseguir las 12 perlas, nota que hay una variable dedicada a comparar las 12 perlas y otra a las simulaciones que se corren, ambas modificables.

### 2. Correr Gráfica de frecuencias y gráfica acumulada
En realidad no es obligatorio pero lo hice para darme una idea de los datos con los que estabamos tratando.

### 3. Modelador Gompertz/Richards
Cuando ejecutes cualquiera de estos dos se mostraran dos gráficas, una seguida de la otra. La primera es el propio ajuste y la segunda el error, igualmente en la terminal se imprimen dos cosas, los valores del ajuste y el informe de errores.

### 4. Graficador
Aquí junté las funciones según los parámetros que obtuve y los comparé dentro de una misma gráfica, literalmente hay dos funciones que puedes modificar :D
