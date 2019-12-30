import numpy as np

#Introducimos por teclado las dimensiones de la matriz y la generamos con valores aleatorios
rows=input("Introduce el numero de filas:")
cols=input("Introduce el numero de columnas:")
m=np.random.rand(int(rows), int(cols))
m=np.array(m)
print(m)
#Imprimimos el valor maximo y el minimo de la matriz
print("El maximo valor de la matriz es ", m.max())
print("El minimo valor de la matriz es ", m.min())

#Calculamos el producto escalar para dos filas o dos columnas
aux=input("Introduzca 0 si quiere que se use dos filas para el producto escalar y un valor distinto de cero para usar dos columnas:")
if(int(aux)==0):
    fila1=input("Introduce el numero de la primera fila a usar:")
    fila2=input("Introduce el numero de la segunda fila a usar:")
    if(int(fila1)<int(rows) and int(fila2)<int(rows) and int(fila1)>=0 and int(fila2)>=0):
        v1=m[int(fila1)]
        v2=m[int(fila2)]
        producto_escalar=np.sum(v1*v2)
        modv1=np.sqrt(sum(v1*v1))
        modv2=np.sqrt(sum(v2*v2))
        aux2=producto_escalar/(modv1*modv2)
        angulo=np.arccos(aux)
        print("El angulo formado por los dos vectores fila es: ",(angulo*180)/3.141592653589793," grados")
    else:
        print("Indices de fila de la matriz no validos")
else:
    col1=input("Introduce el numero de la primera columna a usar:")
    col2=input("Introduce el numero de la segunda columna a usar:")
    if(int(col1)<int(cols) and int(col2)<int(cols) and int(col1)>=0 and int(col2)>=0):
        v1=m[:,int(col1)]
        v2=m[:,int(col2)]
        producto_escalar=np.sum(v1*v2)
        modv1=np.sqrt(sum(v1*v1))
        modv2=np.sqrt(sum(v2*v2))
        aux2=producto_escalar/(modv1*modv2)
        angulo=np.arccos(aux)
        print("El angulo formado por los dos vectores columna es: ",(angulo*180)/3.141592653589793," grados")       
    else:
        print("Indices de columna de la matriz no validos")