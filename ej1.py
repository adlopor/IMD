#ej1.py
#Implemente mediante un programa Python la asignación del reparto de escaños de una circunscripción electoral usando la Ley D’Hondt.
#Los datos se pueden introducir por teclado o leerse desde un fichero.

#Librerias importadas
import numpy as np
import math

#Se introducen los parámetros de entrada:
#Primero introducimos el número de partidos políticos que se presentan a las elecciones
partidos= int(input("Introduce el número de partidos políticos que se han presentado a las elecciones municipales: "))
while (partidos<=0):
    print("Valor incorrecto. Se ha de presentar al menos un partido...")
    partidos= int(input("Vuelva a introducir el número de partidos políticos que se han presentado a las elecciones municipales: "))

#Luego, introducimos el número de escaños se reparten en las elecciones
escanos= int(input("Introduce el número de escaños que hay en el municipio: "))
while (escanos<=0):
    print('Valor incorrecto. La localidad debe tener representación...')
    escanos= int(input('Vuelva a introducir el número de escaños que hay en el municipio: '))

#Creamos el array de votos obtenidos de forma aleatoria
votos=np.random.randint(101, size=(partidos))
print("votos iniciales: ",votos)


#Luego generamos la matriz de la Ley d'Hont a través del los votos obtenidos en el escrutinio.
m= np.zeros(shape=(partidos,escanos))
print("Matriz ley d'Hont:")

for i in range(len(m)):
    for j in range(len(m[i])):
        m[i][j]=math.ceil(votos[i]/(j+1))

m.astype(int)
print(m)

#Ahora calculamos los N-máximos valores de la matriz, siendo N el número de escaños.


mAux = np.copy(m)
vectormaximos = np.zeros(shape=(1,escanos))

pos = [0,0]

maximo= mAux[0][0]
print (maximo)

#for k in range (escanos):
#    for i in range(len(mAux)):
#        for j in range(len(mAux[i])):
#            if ( maximo < mAux[i][j] ):
#                pos = [i,j]
                
#    maximo=mAux(pos)
#    vectormaximos[1][k]=maximo
    
    
            





