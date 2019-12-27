import numpy as np

matriz=np.array([],dtype=int)
rows=input("Introduce el numero de filas:")
cols=input("Introduce el numero de columnas:")
z=0
for i in range(0,int(rows)):
    aux=[]
    for j in range(0,int(cols)):
        print("Introduce el valor de ",i,",",j,":")
        value=input() 
        aux.append(int(value))
    if(z==0):
        matriz=aux
        z=1
    else:
        matriz=np.vstack([matriz,aux])   
print(matriz) 

a,b=np.unique(matriz, return_counts=True)
print("La moda de la matriz es ",a[np.argmax(b)])
print("La media de la matriz es ",matriz.mean())