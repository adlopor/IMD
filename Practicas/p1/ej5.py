import numpy as np

#Leemos la matriz del fichero
file=input("Introduce el fichero que contiene la matriz:")
file_object  = open(file, "r")

matriz=np.array([line.split(' ') for line in file_object]);
for i,item in enumerate(matriz):
    for j,item2 in enumerate(item):
        #Borra los espacios del string para que no de problemas
        matriz[i,j]=item2.rstrip()

#Comprobamos que sea cuadrada
matriz = matriz.astype('float') 
if(matriz[:,0].size!=matriz[0].size):
    print("La matriz no es invertible")
    exit()

#Comprobamos que la matriz sea invertible (cuando el determinante es distinto de cero), se pone un numero chico por el error cometido al redondear
det=np.linalg.det(matriz);
if(1e-9>abs(det)):
    print("La matriz no es invertible")
    exit()

#Tras comprobar que es invertible, se calcula la inversa
matriz_inv=matriz.copy()
for i in range(matriz[0].size):
    for j in range(matriz[0].size):
        aux=matriz;
        aux=np.delete(aux,j,axis=1)
        aux=np.delete(aux,i,axis=0)
        matriz_inv[i,j]=((-1)**(i+j))*np.linalg.det(aux)

matriz_inv=matriz_inv.transpose()
matriz_inv=matriz_inv/det
print("La matriz inversa es: ")
print(matriz_inv)
print("El producto matricial es (deberia de dar la matriz identidad de tamano 4):")
print(abs(np.around(np.matmul(matriz_inv,matriz))))