import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

data = pd.read_csv('AMD_clustering_0.csv',  # Cargamos el csv
                       delimiter=',',
                       header=0)

data = data.sample(frac=0.01) 

data.head();
print("Dataset shape:", data.shape);
data.isnull().any().any();

modelo=DBSCAN(eps=0.05, min_samples=27)
clusterDBSCAN=data.copy()
clusterDBSCAN['dbscan']=modelo.fit_predict(clusterDBSCAN.to_numpy())
sb.pairplot(clusterDBSCAN, hue='dbscan')
plt.show()