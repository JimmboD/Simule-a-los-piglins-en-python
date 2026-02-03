import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Carga los datos desde el archivo CSV
df = pd.read_csv(os.path.join(os.getcwd(), r"SimuladorPiglins\resultados.csv"))
datos = df['# Intentos'].sort_values() 

y = np.arange(1, len(datos) + 1) / len(datos) #Escala
x = datos

# Configuaciones de la grafica
plt.figure(figsize=(10, 6))
plt.plot(x, y, marker='.', linestyle='none', color='red', markersize=2)

plt.title('Probabilidad de 12 perlas', fontsize=16)
plt.xlabel('Lingotes intercambiados', fontsize=12)
plt.ylabel('Probabilidad Acumulada', fontsize=12)
plt.grid(True, alpha=0.3)

plt.show()