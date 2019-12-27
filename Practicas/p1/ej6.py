import numpy as np
file=input("Introduce el nombre del fichero que contiene el sistema:")
file_object  = open(file, "r")

matriz=np.array([line.split(' ') for line in file_object]);
for i,item in enumerate(matriz):
    for j,item2 in enumerate(item):
        matriz[i,j]=item2.rstrip()
matriz = matriz.astype('float') 
if(matriz[:,0].size+1!=matriz[0].size):
    print("Sistema incompatible")
    exit()
matdep=matriz[:matriz[:,0].size,:matriz[:,0].size]
matind=matriz[:,matriz[0].size-1]

determind=np.linalg.det(matdep)
if(abs(determind)<1e-6):
    print("Sistema incompatible")
    exit()

solucion=np.array([])

for i in range(matriz[:,0].size):
    mataux=matdep.copy()
    mataux[:,i]=matind
    solucion=np.append(solucion,[np.linalg.det(mataux)/determind])
print(solucion)