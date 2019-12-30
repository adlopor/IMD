from sklearn import preprocessing
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.io import arff

def ej9function(file, classname):
	data = arff.loadarff(file)
	df = pd.DataFrame(data[0])
	pd.plotting.parallel_coordinates(df, classname)
	plt.show()

ej9function('Datasets/iris.arff', 'class')
ej9function('Datasets/diabetes.arff', 'class')
ej9function('Datasets/vote.arff', 'Class')