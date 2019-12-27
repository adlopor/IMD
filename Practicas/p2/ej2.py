import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

fig, axs = plt.subplots(ncols=2,sharex=True)
nombre_fichero="basesDeDatos/diabetes.csv"
data = pd.read_csv(nombre_fichero) 
data.groupby("class").plas.hist(alpha=0.4, ax=axs[1])
axs[1].set_title('Histograma sin normalizar')
data.groupby("class").plas.hist(alpha=0.4,density=True, ax=axs[0])
axs[0].set_title('Histograma normalizado')
plt.xlabel("Glucosa plasmática tras el test")
plt.ylabel("Frecuencia")
plt.legend(["tested_negative","tested_positive"])
plt.show()