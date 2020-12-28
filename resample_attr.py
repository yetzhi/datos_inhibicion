import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
import csv

df = pd.read_csv("g2-r7_attr.csv")
#Ver cantidad de Ejemplos por cada clase
#print (df['class'].value_counts())

#Verificación de valores nulos
#sns.heatmap(data=df.isnull(), yticklabels=False, cbar=False, cmap='viridis')

#
#datos = df.isna().sum()
#datos2 = df.isna().any()
#print (datos2)

#balance de clases
#X,y = reducir_parametros(X,y)
df_max = df[df.clase==0]
df_min = df[df.clase==1]
df_min_balance = resample(df_min, replace=True, n_samples=46, random_state=0)
df = pd.concat([df_max, df_min_balance])

df.to_csv('g2-r7_attr_resample.csv', index=False)
#print (df['clase'].value_counts())
