import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.io import arff

#Hacemos una funcion a la que pasamos el nombre del fichero y ahorramos codigo
def ej2function(file):
    #Cargamos archivo y generamos el histograma que se muestra por pantalla.
	data = arff.loadarff(file)
	df = pd.DataFrame(data[0])
	df.plot.hist(alpha=0.5)
	plt.show()

ej2function('Datasets/iris.arff')
ej2function('Datasets/diabetes.arff')
ej2function('Datasets/vote.arff')