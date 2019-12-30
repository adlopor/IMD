import numpy as np

#Introducimos por teclado las dimensiones de la matriz
matriz=np.array([],dtype=int)
rows=input("Introduzca el numero de filas:")
cols=input("Introduzca el numero de columnas:")

#Igual que en el ej3.py
z=0
for i in range(0,int(rows)):
    aux=[]
    for j in range(0,int(cols)):
        print("Introduzca el valor de ",i,",",j,":")
        value=input() 
        aux.append(int(value))
    if(z==0):
        matriz=aux
        z=1
    else:
        matriz=np.vstack([matriz,aux])   
print(matriz) 

#Calculamos la media y la moda de la matriz
a,b=np.unique(matriz, return_counts=True)
print("La moda de la matriz es ",a[np.argmax(b)])
print("La media de la matriz es ",matriz.mean())