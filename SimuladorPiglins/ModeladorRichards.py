import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

df = pd.read_csv(os.path.join(os.getcwd(), r"SimuladorPiglins\resultados.csv"))

columna = df.columns[0] 

tabla = df[columna].value_counts().sort_index().reset_index()
tabla.columns = ['Intento', 'Frecuencia'] # Renombramos para claridad

tabla['PDF'] = tabla['Frecuencia'] / tabla['Frecuencia'].sum()

tabla['CDF'] = tabla['PDF'].cumsum()

print(tabla.head(20))

# Función de Richards (Sigmoide Asimétrica)
def richards_curve(x, L, k, x0, s):
    return L / ((1 + np.exp(-k * (x - x0)))**s)

# Tus datos
x_datos = tabla['Intento'].values
y_datos = tabla['CDF'].values

# CONFIGURACIÓN ROBUSTA
# 1. Pistas iniciales más suaves (k=0.1 ayuda mucho)
p0_robusto = [1.0, 0.1, np.mean(x_datos), 1.0]

# 2. Límites (Bounds)
# L: entre 0.99 y 1.01 (es probabilidad)
# k: entre 0.001 y 5 (pendiente positiva)
# s: entre 0.01 y 20 (asimetría razonable)
bounds_robustos = ([0.99, 0.001, -np.inf, 0.01], [1.01, 5.0, np.inf, 200000.0])

try:
    popt, pcov = curve_fit(
        richards_curve, 
        x_datos, 
        y_datos, 
        p0=p0_robusto, 
        bounds=bounds_robustos,
        method='trf',      # <--- CAMBIO CLAVE: Algoritmo más estable
        maxfev=20000       # <--- CAMBIO CLAVE: 20k intentos
    )

    L_opt, k_opt, x0_opt, s_opt = popt
    print(f"¡Ajuste exitoso!\n L={L_opt:.3f}, k={k_opt:.3f}, x0={x0_opt:.1f}, s={s_opt:.3f}")
    
    # Graficar
    x_plot = np.linspace(min(x_datos), max(x_datos), 200)
    plt.figure(figsize=(10,6))
    plt.scatter(x_datos, y_datos, color='black', s=10, label='Datos')
    plt.plot(x_plot, richards_curve(x_plot, *popt), color='red', lw=2, label='Richards (Ajustada)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

except Exception as e:
    print("Error crítico:", e)
    print("Prueba la 'Solución Nuclear' (Skew Normal) si esto falla.")


# --- CÁLCULO DE ERRORES (BONDAD DE AJUSTE) ---

# 1. Calcular los valores que predice tu curva con los parámetros optimizados
y_pred = richards_curve(x_datos, *popt)

# 2. Calcular los Residuos (Diferencia Realidad - Predicción)
residuos = y_datos - y_pred

# 3. Métricas de Error
# MSE: Error Cuadrático Medio
mse = np.mean(residuos**2)
# RMSE: Raíz del Error Cuadrático Medio (está en las mismas unidades que tu Y, o sea probabilidad)
rmse = np.sqrt(mse)
# R^2: Coeficiente de Determinación (qué tan bien explica el modelo a los datos)
ss_res = np.sum(residuos**2)
ss_tot = np.sum((y_datos - np.mean(y_datos))**2)
r_squared = 1 - (ss_res / ss_tot)

print("-" * 30)
print("REPORTE DE ERRORES DEL AJUSTE")
print("-" * 30)
print(f"Error Promedio (RMSE): {rmse:.5f}")
print(f"Calidad del Ajuste (R^2): {r_squared:.5f}")
print("-" * 30)

if r_squared > 0.99:
    print(">> AJUSTE PERFECTO: La curva pasa casi exactamente por los datos.")
elif r_squared > 0.95:
    print(">> AJUSTE BUENO: Hay pequeñas desviaciones pero el modelo es útil.")
else:
    print(">> AJUSTE POBRE: La curva no representa bien la realidad.")

# 4. GRAFICAR EL ERROR (RESIDUOS)
# Si la curva es buena, esta gráfica debe verse como "ruido" aleatorio alrededor del cero.
# Si ves una forma de "U" o patrones raros, tu modelo (Richards) no fue el correcto.
plt.figure(figsize=(10, 4))
plt.scatter(x_datos, residuos, color='red', s=5, alpha=0.5)
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.title('Gráfica de Residuos (Diferencia entre Datos y Curva)')
plt.ylabel('Error (Residual)')
plt.xlabel('Cantidad de Intentos')
plt.grid(True, alpha=0.3)
plt.show()