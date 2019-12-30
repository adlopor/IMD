from sklearn import preprocessing
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.io import arff

def ej8function(file):
	data = arff.loadarff(file)
	df = pd.DataFrame(data[0])
	corr = df.corr()
	sns.heatmap(corr, xticklabels=corr.columns,yticklabels=corr.columns)
	plt.show()


ej8function('Datasets/iris.arff')
ej8function('Datasets/diabetes.arff')
ej8function('Datasets/vote.arff')