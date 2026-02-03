import numpy as np


# Esta función simula el tradeo con Piglins hasta obtener 12 perlas de ender
def simular_tradeo():
    elementos = ["Perla", "Otro"]
    probabilidades = [10/459, 449/459] # Probabilidades ajustadas a los piglins y el drop de perlas

    rolls =[2,3,4] # Posibles cantidades de perlas obtenidas en un tradeo exitoso

    loot=0
    
    resultado = np.random.choice(elementos, p=probabilidades)

    if resultado != "Otro":
        loot=int(np.random.choice(rolls))
    else:
        loot=0

    return(loot)

# Rectifica que el tradeo se repita hasta obtener 12 perlas
def contador():

    Contador=0

    for i in range(9999):#máximo 9999 intentos
        loot=simular_tradeo()
        
        Contador+=loot

        if Contador >= 12:
            return i+1

# Genera datos y los guarda en un archivo CSV   
def generar_datos(a):
    resultados=[]

    for i in range(a):
        intentos=contador()
        resultados.append(intentos)

        # Mensaje de progreso
        if i % 1000 == 0:
            print(int(i/a*100),"% completado")


    np.savetxt(r"C:\Users\jimbo\Desktop\Programación\Escuela\SimuladorPiglins\resultados.csv", resultados, delimiter=",", header="Intentos", fmt='%d')
    print("Archivo generado.")

# Principal
if __name__ == "__main__":

    generar_datos(99999)#Número de simulaciones a correr