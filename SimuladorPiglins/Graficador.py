import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import pandas as pd
import os

def gompertz_curve(x):
    return np.exp(-np.exp(-0.013 * (x - 154.4)))

def Richardson_curve(x):
    return 1.003 / (1 + np.exp(-0.012 * (x + 577.6)))**9015.066

df = pd.read_csv(os.path.join(os.getcwd(), r"SimuladorPiglins\resultados.csv"))

columna = df.columns[0] 

tabla = df[columna].value_counts().sort_index().reset_index()
tabla.columns = ['Intento', 'Frecuencia'] # Renombramos para claridad

tabla['PDF'] = tabla['Frecuencia'] / tabla['Frecuencia'].sum()

tabla['CDF'] = tabla['PDF'].cumsum()


x = np.linspace(0, 800, 1600)
y_gompertz = gompertz_curve(x)
y_richardson = Richardson_curve(x)
y_datos = tabla['CDF'].values
plt.figure(figsize=(12, 6))
plt.plot(x, y_gompertz, label='Curva de Gompertz', color='blue')
plt.plot(x, y_richardson, label='Curva de Richards', color='red')
plt.scatter(tabla['Intento'], y_datos, color='black', s=10, label='Datos Reales')
plt.title('Comparación de Curvas: Gompertz vs Richards')
plt.xlabel('Lingotes tradeados')
plt.ylabel('Probabilidad')
plt.legend()

ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MultipleLocator(30)) 
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1)) 
plt.grid(True, which='major', color='gray', linestyle='--', alpha=0.5)


plt.grid(True, alpha=0.3)
plt.show()