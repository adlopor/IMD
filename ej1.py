

#Se introducen los parámetros de entrada.

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
#np.random.randint(101, size=(partidos))

#Luego generamos la matriz de la Ley d'Hont a través del los votos obtenidos en el escrutinio y guardamos la matriz en un fichero.
#np.random.randint(101, size=(2,4))