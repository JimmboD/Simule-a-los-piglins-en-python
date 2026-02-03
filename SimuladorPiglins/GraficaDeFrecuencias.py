import pandas as pd
import matplotlib.pyplot as plt
import os

# Carga los datos desde el archivo CSV
df = pd.read_csv(os.path.join(os.getcwd(), r"SimuladorPiglins\resultados.csv"))
datos = df['# Intentos']

# Configuraciones de la gráfica
plt.figure(figsize=(10, 6))

plt.hist(datos, bins=range(datos.min(), datos.max() + 2), 
         color="BLUE")

plt.title('Gráfica de freciencias', fontsize=16)
plt.xlabel('Tradeos', fontsize=12)
plt.ylabel('Frecuencia', fontsize=12)
plt.grid(alpha=0.5)

plt.show()