import numpy as np

rows=input("Introduce el numero de filas y columnas:")
rows=int(rows)
matriz=np.array([],dtype=float)
z=0
for i in range(0,rows):
    aux=[]
    for j in range(0,rows):
        print("Introduce el valor de ",i,",",j,":")
        value=input() 
        aux.append(float(value))
    if(z==0):
        matriz=aux
        z=1
    else:
        matriz=np.vstack([matriz,aux])   
print(matriz) 
r=0       
for i in matriz.max(axis=1):
    print("El maximo de la fila ",r," es ",i)
    r=r+1  
r=0  
for i in matriz.max(axis=0):
    print("El maximo de la columna ",r," es ",i)
    r=r+1             
print("El determinante de la matriz es ",np.linalg.det(matriz))  
print("El rango de la matriz es ", np.linalg.matrix_rank(matriz))