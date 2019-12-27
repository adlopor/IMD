import numpy as np

rows=input("Introduce el numero de filas:")
cols=input("Introduce el numero de columnas:")
randmat=np.random.rand(int(rows), int(cols))
randmat=np.array(randmat)
print(randmat)
print("El maximo valor de la matriz es ", randmat.max())
print("El minimo valor de la matriz es ", randmat.min())

filaocol=input("Introduce 1 si quiere que se use dos filas y cualquier otra cosa para columnas:")
if(int(filaocol)==1):
    fila1=input("Introduce la primera fila:")
    fila2=input("Introduce la segunda fila:")
    if(int(fila1)<int(rows) and int(fila2)<int(rows) and int(fila1)>=0 and int(fila2)>=0):
        u=randmat[int(fila1)]
        v=randmat[int(fila2)]
        dotpr=np.sum(u*v)
        modu=np.sqrt(sum(u*u))
        modv=np.sqrt(sum(v*v))
        aux=dotpr/(modu*modv)
        angulo=np.arccos(aux)
        print("El angulo formado por los dos vectores es ",(angulo*180)/3.141592653589793,"º")
    else:
        print("Indices no validos")
else:
    col1=input("Introduce la primera columna:")
    col2=input("Introduce la segunda columna:")
    if(int(col1)<int(cols) and int(col2)<int(cols) and int(col1)>=0 and int(col2)>=0):
        u=randmat[:,int(col1)]
        v=randmat[:,int(col2)]
        dotpr=np.sum(u*v)
        modu=np.sqrt(sum(u*u))
        modv=np.sqrt(sum(v*v))
        aux=dotpr/(modu*modv)
        angulo=np.arccos(aux)
        print("El angulo formado por los dos vectores es ",(angulo*180)/3.141592653589793,"º")       
    else:
        print("Indices no validos")