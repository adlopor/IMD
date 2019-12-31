
import sys, os, pydotplus
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from sklearn import neighbors, preprocessing, metrics
from sklearn.tree import export_graphviz
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics.cluster import homogeneity_score
from sklearn.model_selection import train_test_split
from sklearn.externals.six import StringIO
from sklearn.multiclass import OutputCodeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from IPython.display import Image
from pandas.api.types import is_numeric_dtype

#Parte del codigo reutilizado de la practica 3
def preprocesar_datos(filename):
	ds = pd.read_csv(filename)
	df = pd.DataFrame(ds)
	d = df.iloc[:, df.columns != 'class']
	objetivo = pd.factorize(df['class'])[0]
	trainInputs, testInputs, trainOutputs, testOutputs = train_test_split(d, objetivo, test_size=0.7)
	return d, trainInputs, trainOutputs, testInputs, testOutputs

#Funcion para calcular la matriz de incidencias
def calcularMatrizIncidencias(resultados):
	matriz = np.zeros((resultados.labels_.size, resultados.labels_.size))
	for i in resultados.labels_:
		for j in resultados.labels_:
			if i == j:
				matriz[i][j] = 1
			else:
				matriz[i][j] = 0
	return matriz

#author: Erableto
#Funcion para calcular la matriz de distacias
def calcularMatrizDistancias(matrizIncidencias):
	X = scipy.spatial.distance.pdist(matrizIncidencias, metric='euclidean')
	matriz = scipy.spatial.distance.squareform(X, force='no', checks=True)
	return matriz

#Funcion que hace clustering con KMeans
def clusteringKMeans(d, testInputs, testOutputs, graphname, clusteres):
	print("\n\t[" + str(graphname) + "]")
	kmeans=KMeans(n_clusters = clusteres)
	resultados=kmeans.fit(d)
	print("\n\t\tCluster: " + str(resultados.labels_))
	homogeneidad = homogeneity_score(testOutputs, kmeans.predict(testInputs, testOutputs))
	print("\n\t\tHomogeneidad: " + str(homogeneidad))
	print("\n\t\tMedida de evaluacion no supervisada: " + str(metrics.adjusted_mutual_info_score(testOutputs, kmeans.predict(testInputs,testOutputs))))
	mI = calcularMatrizIncidencias(resultados)
	mD = calcularMatrizDistancias(mI)
	print("\n\t\tCorrelación entre matrices: " + str(np.corrcoef(mI, mD)))
	print("\n")
	return homogeneidad

#Funcion que hace clustering con singleLink
def clusteringSingleLink(d, testInputs, testOutputs, graphname, clusteres):
	print("\n\t[" + str(graphname) + "]")
	singlelink=AgglomerativeClustering(n_clusters = clusteres, linkage = 'single')
	resultados=singlelink.fit(d)
	print("\n\t\tCluster: " + str(resultados.labels_))
	homogeneidad = homogeneity_score(testOutputs, singlelink.fit_predict(testInputs, testOutputs))
	print("\n\t\tHomogeneidad: " + str(homogeneidad))
	print("\n\n")
	return homogeneidad

#Funcion que hace clustering con completeLink
def clusteringCompleteLink(d, testInputs, testOutputs, graphname, clusteres):
	print("\n\t[" + str(graphname) + "]")
	completelink=AgglomerativeClustering(n_clusters = clusteres, linkage = 'complete')
	resultados=completelink.fit(d)
	print("\n\t\tCluster: " + str(resultados.labels_))
	homogeneidad = homogeneity_score(testOutputs, completelink.fit_predict(testInputs, testOutputs))
	print("\n\t\tHomogeneidad: " + str(homogeneidad))
	print("\n\n")
	return homogeneidad

#Funcion que hace clustering con averageLink
def clusteringAverageLink(d, testInputs, testOutputs, graphname, clusteres):
	print("\n\t[" + str(graphname) + "]")
	averagelink=AgglomerativeClustering(n_clusters = clusteres, linkage = 'average')
	resultados=averagelink.fit(d)
	print("\n\t\tClúster: " + str(resultados.labels_))
	homogeneidad = homogeneity_score(testOutputs, averagelink.fit_predict(testInputs, testOutputs))
	print("\n\t\tHomogeneidad: " + str(homogeneidad))
	print("\n\n")
	return homogeneidad

#Funcion que calcula con cuántos clusteres clasifica mejor el calsificador
def clusteringKMeansRendimiento(d, testInputs, testOutputs, graphname):
	bestKNN = 0
	bestCKNN = 0
	bestSingleLink = 0
	bestCSingleLink = 0
	bestCompleteLink = 0
	bestCCompleteLink = 0
	bestAverageLink = 0
	bestCAverageLink = 0

	for clusteres in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]:
		homogeneidadKNN = clusteringKMeans(d, testInputs, testOutputs, graphname, clusteres)
		if homogeneidadKNN > bestKNN:
			bestKNN = homogeneidadKNN
			bestCKNN = clusteres
		
		homogeneidadSingle = clusteringSingleLink(d, testInputs, testOutputs, graphname, clusteres)
		if homogeneidadSingle > bestSingleLink:
			bestSingleLink=homogeneidadSingle
			bestCSingleLink = clusteres
		
		homogeneidadComplete = clusteringCompleteLink(d, testInputs, testOutputs, graphname, clusteres)
		if homogeneidadComplete > bestCompleteLink:
			bestCompleteLink=homogeneidadComplete
			bestCCompleteLink = clusteres
		
		homogeneidadAverage = clusteringAverageLink(d, testInputs, testOutputs, graphname, clusteres)
		if homogeneidadAverage > bestAverageLink:
			bestAverageLink=homogeneidadAverage
			bestCAverageLink = clusteres

	print("\n[RESULTADOS PARA " + str(graphname) + "]")
	print("\n\t{bestKNN, bestCKNN, bestSingleLink, bestCSingleLink, bestCompleteLink, bestCCompleteLink, bestAverageLink, bestCAverageLink}")
	print("\n\t" + str(bestKNN) + " " + str(bestCKNN) + " " + str(bestSingleLink) + " " + str(bestCSingleLink) + " " + str(bestCompleteLink) + " " + str(bestCCompleteLink) + " " + str(bestAverageLink) + " " + str(bestCAverageLink))
	return bestKNN, bestCKNN, bestSingleLink, bestCSingleLink, bestCompleteLink, bestCCompleteLink, bestAverageLink, bestCAverageLink

d_iris, trI_iris, trO_iris, teI_iris, teO_iris = preprocesar_datos('Datasets_no_class/iris.csv')
d_contact_lenses, trI_contact_lenses, trO_contact_lenses, teI_contact_lenses, teO_contact_lenses = preprocesar_datos('Datasets_no_class/contact_lenses.csv')
d_soybean, trI_soybean, trO_soybean, teI_soybean, teO_soybean = preprocesar_datos('Datasets_no_class/soybean.csv')
d_glass, trI_glass, trO_glass, teI_glass, teO_glass = preprocesar_datos('Datasets_no_class/glass.csv')
d_car, trI_car, trO_car, teI_car, teO_car = preprocesar_datos('Datasets_no_class/car.csv')


print("\n  [clusteringKMeans]")

clusteringKMeans(d_iris, teI_iris, teO_iris, 'P6_iris_clusteringKMeans.svg', 5)
clusteringKMeans(d_contact_lenses, teI_contact_lenses, teO_contact_lenses, 'P6_contact_lenses_clusteringKMeans.svg', 5)
clusteringKMeans(d_soybean, teI_soybean, teO_soybean, 'P6_soybean_clusteringKMeans.svg', 5)
clusteringKMeans(d_glass, teI_glass, teO_glass, 'P6_glass_clusteringKMeans.svg', 5)
clusteringKMeans(d_car, teI_car, teO_car, 'P6_car_clusteringKMeans.svg', 5)


print("\n  [clusteringSingleLink]")

clusteringSingleLink(d_iris, teI_iris, teO_iris, 'P6_iris_clusteringSingleLink.svg', 5)
clusteringSingleLink(d_contact_lenses, teI_contact_lenses, teO_contact_lenses, 'P6_contact_lenses_clusteringSingleLink.svg', 5)
clusteringSingleLink(d_soybean, teI_soybean, teO_soybean, 'P6_soybean_clusteringSingleLink.svg', 5)
clusteringSingleLink(d_glass, teI_glass, teO_glass, 'P6_glass_clusteringSingleLink.svg', 5)
clusteringSingleLink(d_car, teI_car, teO_car, 'P6_car_clusteringSingleLink.svg', 5)


print("\n  [clusteringCompleteLink]")

clusteringCompleteLink(d_iris, teI_iris, teO_iris, 'P6_iris_clusteringCompleteLink.svg', 5)
clusteringCompleteLink(d_contact_lenses, teI_contact_lenses, teO_contact_lenses, 'P6_contact_lenses_clusteringCompleteLink.svg', 5)
clusteringCompleteLink(d_soybean, teI_soybean, teO_soybean, 'P6_soybean_clusteringCompleteLink.svg', 5)
clusteringCompleteLink(d_glass, teI_glass, teO_glass, 'P6_glass_clusteringCompleteLink.svg', 5)
clusteringCompleteLink(d_car, teI_car, teO_car, 'P6_car_clusteringCompleteLink.svg', 5)


print("\n  [clusteringAverageLink]")

clusteringAverageLink(d_iris, teI_iris, teO_iris, 'P6_iris_clusteringAverageLink.svg', 5)
clusteringAverageLink(d_contact_lenses, teI_contact_lenses, teO_contact_lenses, 'P6_contact_lenses_clusteringAverageLink.svg', 5)
clusteringAverageLink(d_soybean, teI_soybean, teO_soybean, 'P6_soybean_clusteringAverageLink.svg', 5)
clusteringAverageLink(d_glass, teI_glass, teO_glass, 'P6_glass_clusteringAverageLink.svg', 5)
clusteringAverageLink(d_car, teI_car, teO_car, 'P6_car_clusteringAverageLink.svg', 5)


print("\n  [clusteringKMeansRendimiento]")

clusteringKMeansRendimiento(d_iris, teI_iris, teO_iris, 'P6_iris_clusteringKMeansRendimiento.svg')
clusteringKMeansRendimiento(d_contact_lenses, teI_contact_lenses, teO_contact_lenses, 'P6_contact_lenses_clusteringKMeansRendimiento.svg')
clusteringKMeansRendimiento(d_soybean, teI_soybean, teO_soybean, 'P6_soybean_clusteringKMeansRendimiento.svg')
clusteringKMeansRendimiento(d_glass, teI_glass, teO_glass, 'P6_glass_clusteringKMeansRendimiento.svg')
clusteringKMeansRendimiento(d_car, teI_car, teO_car, 'P6_car_clusteringKMeansRendimiento.svg')


print("\n  [clusters = clases][clusteringKMeans]")

clusteringKMeans(d_iris, teI_iris, teO_iris, 'P6_iris_clusteringKMeans.svg', 3)
clusteringKMeans(d_contact_lenses, teI_contact_lenses, teO_contact_lenses, 'P6_contact_lenses_clusteringKMeans.svg', 3)
clusteringKMeans(d_soybean, teI_soybean, teO_soybean, 'P6_soybean_clusteringKMeans.svg', 19)
clusteringKMeans(d_glass, teI_glass, teO_glass, 'P6_glass_clusteringKMeans.svg', 7)
clusteringKMeans(d_car, teI_car, teO_car, 'P6_car_clusteringKMeans.svg', 4)


print("\n  [clusters = clases][clusteringSingleLink]")

clusteringSingleLink(d_iris, teI_iris, teO_iris, 'P6_iris_clusteringSingleLink.svg', 3)
clusteringSingleLink(d_contact_lenses, teI_contact_lenses, teO_contact_lenses, 'P6_contact_lenses_clusteringSingleLink.svg', 3)
clusteringSingleLink(d_soybean, teI_soybean, teO_soybean, 'P6_soybean_clusteringSingleLink.svg', 19)
clusteringSingleLink(d_glass, teI_glass, teO_glass, 'P6_glass_clusteringSingleLink.svg', 7)
clusteringSingleLink(d_car, teI_car, teO_car, 'P6_car_clusteringSingleLink.svg', 4)


print("\n  [clusters = clases][clusteringCompleteLink]")

clusteringCompleteLink(d_iris, teI_iris, teO_iris, 'P6_iris_clusteringCompleteLink.svg', 3)
clusteringCompleteLink(d_contact_lenses, teI_contact_lenses, teO_contact_lenses, 'P6_contact_lenses_clusteringCompleteLink.svg', 3)
clusteringCompleteLink(d_soybean, teI_soybean, teO_soybean, 'P6_soybean_clusteringCompleteLink.svg', 19)
clusteringCompleteLink(d_glass, teI_glass, teO_glass, 'P6_glass_clusteringCompleteLink.svg', 7)
clusteringCompleteLink(d_car, teI_car, teO_car, 'P6_car_clusteringCompleteLink.svg', 4)


print("\n  [clusters = clases][clusteringAverageLink]")

clusteringAverageLink(d_iris, teI_iris, teO_iris, 'P6_iris_clusteringAverageLink.svg', 3)
clusteringAverageLink(d_contact_lenses, teI_contact_lenses, teO_contact_lenses, 'P6_contact_lenses_clusteringAverageLink.svg', 3)
clusteringAverageLink(d_soybean, teI_soybean, teO_soybean, 'P6_soybean_clusteringAverageLink.svg', 19)
clusteringAverageLink(d_glass, teI_glass, teO_glass, 'P6_glass_clusteringAverageLink.svg', 7)
clusteringAverageLink(d_car, teI_car, teO_car, 'P6_car_clusteringAverageLink.svg', 4)
