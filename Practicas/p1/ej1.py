import numpy as np

#Leemos el fichero que contiene los datos y lo almacenamos en un array
file=input("Introduce el nombre del fichero que contiene los votos:")
file_object  = open(file, "r")
datos_fichero=np.array([line.split(' ') for line in file_object]);
for i in datos_fichero:
    i[1]=i[1][:-1]
numEscanos=input("Introduce el numero de escagnos a introducir:")
matriz=np.array([])
k=0
for i in datos_fichero:
    v=[]
    for j in range(1,int(numEscanos)+1):
        v.append(int(int(i[1])/int(j)))
    if(k==0):
        matriz=v;
        k=1
    else:
        matriz=np.vstack([matriz,v])
        
#Calculamos los escagnos que se lleva cada partido
numEscanos_finales=np.array([0 for i in range(0,np.size(datos_fichero,0))])
aux=numEscanos
while(aux!=0):
    numEscanos_finales[int(matriz.argmax()/numEscanos)]=numEscanos_finales[int(matriz.argmax()/numEscanos)]+1;
    np.put(matriz,matriz.argmax(),0)
    aux=aux-1;
for i in range(0,np.size(datos_fichero,0)):
    print("El partido ",datos_fichero[i][0]," ha sacado ",numEscanos_finales[i]," escagnos")