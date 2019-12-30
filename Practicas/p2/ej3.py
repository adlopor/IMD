from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.io import arff

scaler = MinMaxScaler()

def normalizar(file):
	data = arff.loadarff(file)
	df = pd.DataFrame(data[0])
	df = scaler.fit_transform(df.iloc[:, :-1])
	df = pd.DataFrame(df)
	df.plot.hist(alpha=0.5)
	plt.show()

def estandarizar(file):
	data = arff.loadarff(file)
	df = pd.DataFrame(data[0])
	df = df.iloc[:, :-1]
	df = (df-np.mean(df))/np.std(df)
	df.plot.hist(alpha=0.5)
	plt.show()

normalizar('Datasets/iris.arff')
normalizar('Datasets/diabetes.arff')
normalizar('Datasets/vote.arff')

estandarizar('Datasets/iris.arff')
estandarizar('Datasets/diabetes.arff')
estandarizar('Datasets/vote.arff')