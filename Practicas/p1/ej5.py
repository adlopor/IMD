import numpy as np
from _operator import matmul
file=input("Introduce el fichero que contiene la matriz:")
file_object  = open(file, "r")

matriz=np.array([line.split(' ') for line in file_object]);
for i,item in enumerate(matriz):
    for j,item2 in enumerate(item):
        matriz[i,j]=item2.rstrip()

matriz = matriz.astype('float') 
if(matriz[:,0].size!=matriz[0].size):
    print("La matriz no es invertible")
    exit()
    
det=np.linalg.det(matriz);
if(1e-6>abs(det)):
    print("La matriz no es invertible")
    exit()
    
matinv=matriz.copy()
for i in range(matriz[0].size):
    for j in range(matriz[0].size):
            mataux=matriz;
            mataux=np.delete(mataux,j,axis=1)
            mataux=np.delete(mataux,i,axis=0)
            
            matinv[i,j]=((-1)**(i+j))*np.linalg.det(mataux)

matinv=matinv.transpose()
matinv=matinv/det
print("La matriz inversa es: ")
print(matinv)
print("El producto matricial entre ambas matrices es:")
print(abs(np.around(np.matmul(matinv,matriz))))