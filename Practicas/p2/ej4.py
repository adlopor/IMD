import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition

pca = decomposition.PCA(n_components=4)

nombre_fichero="basesDeDatos/aux.csv"
data = pd.read_csv(nombre_fichero) 
pca.fit(data)
data = pca.transform(data)
print(data)
plt.hist(data,alpha=1,cumulative=False,density=True)
plt.legend(["PC1","PC2","PC3","PC4"])
plt.xlabel("Glucosa plasmática tras el test")
plt.ylabel("Frecuencia")
plt.show()