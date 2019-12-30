import numpy as np

#Introducimos por teclado el tamano de la matriz
rows=input("Introduce el numero de filas y columnas, (es una matriz cuadrada):")
rows=int(rows)
matriz=np.array([],dtype=float)
#Rellenamos la matriz
z=0
for i in range(0,rows):
    aux=[]
    for j in range(0,rows):
        print("Introduce el valor de ",i,",",j,":")
        value=input() 
        aux.append(float(value))

    #Para apilar de forma vertical cada nueva fila generada a la matriz, si es la primera fila no hace falta
    if(z==0):
        matriz=aux
        z=1
    else:
        matriz=np.vstack([matriz,aux])

print(matriz) 

#Calculamos el maximo de cada fila y cada columna (recorremos primero por filas y luego por columnas, calculando el maximo de cada una  de ellas)
z=0
for i in matriz.max(axis=1):
    print("El maximo de la fila ",z," es ",i)
    z=z+1  
z=0
for i in matriz.max(axis=0):
    print("El maximo de la columna ",z," es ",i)
    z=z+1

#Calculamos el determinante y el rango de la matriz
print("El determinante de la matriz es ",np.linalg.det(matriz))  
print("El rango de la matriz es ", np.linalg.matrix_rank(matriz))