import pandas as pd
from sklearn import preprocessing
from pkg_resources import parse_version
le = preprocessing.LabelEncoder()
nombre_variables = ['viento', 'v_viento', 'p_ola', 'm_ola','presion','temperatura']
olas = pd.read_csv('train.csv')
le.fit(olas['clase'])
clases_numeros = le.transform(olas['clase'])

olas_df = pd.DataFrame(olas[nombre_variables], columns=nombre_variables)

pd.plotting.scatter_matrix(olas_df, c=clases_numeros,diagonal='kde', figsize=(15, 15),grid=False);