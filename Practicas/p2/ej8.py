import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,10)
cristal = pd.read_csv('glass.csv')
plt.matshow(cristal.corr())

plt.colorbar()