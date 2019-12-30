from sklearn import preprocessing
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.io import arff

def ej5function(file, classname):
	data = arff.loadarff(file)
	df = pd.DataFrame(data[0])
	sns.pairplot(data=df, hue=classname)
	plt.show()

ej5function('Datasets/iris.arff', 'class')
ej5function('Datasets/diabetes.arff', 'class')
ej5function('Datasets/vote.arff', 'Class')