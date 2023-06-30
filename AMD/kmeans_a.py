from sklearn.cluster import KMeans
from sklearn import preprocessing
import sklearn.cluster as cluster
import sklearn.metrics as metrics
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from matplotlib import pyplot as plt
import sklearn.cluster as cluster
import sklearn.metrics as metrics


datos = pd.read_csv('AMD_clustering_0.csv',  # Cargamos el csv
                       delimiter=',',
                       header=0)

datos = datos.sample(frac=0.05) 

print("Dataset: ", datos.shape)


scaler = MinMaxScaler()
scale = scaler.fit_transform(datos[['SS','ST', 'CH', 'ML', 'fC']])
datos_scale = pd.DataFrame(scale, columns = ['SS','ST', 'CH', 'ML', 'fC']);
#print(datos_scale.head(5))

km=KMeans(n_clusters=2, n_init=10)
y_predicted = km.fit_predict(datos[['SS','ST', 'CH', 'ML', 'fC']])
#print(y_predicted)

print(km.cluster_centers_)

datos['Clusters'] = km.labels_

# Encontrar el número de clusters óptimo
for i in range(2,4):
    labels = cluster.KMeans(n_clusters=i,random_state=200, n_init=10).fit(datos_scale).labels_
    print ("Silhouette score for k(clusters) = "+str(i)+" is "
    +str(metrics.silhouette_score(datos_scale,labels,metric="euclidean",sample_size=1000,random_state=200)))
    #El valor óptimo de k es 2

kmeans = cluster.KMeans(n_clusters=2 ,init="k-means++")
kmeans = kmeans.fit(datos[['SS','ST', 'CH', 'ML', 'fC']])

datos['Clusters'] = kmeans.labels_

sns.scatterplot(x="SS", y="ST",hue = 'Clusters',  data=datos, palette='viridis')
plt.savefig("SS_ST.png")
plt.show()

sns.scatterplot(x="SS", y="CH",hue = 'Clusters',  data=datos, palette='viridis')
plt.savefig("SS_CH.png")
plt.show()

sns.scatterplot(x="SS", y="ML",hue = 'Clusters',  data=datos, palette='viridis')
plt.savefig("SS_ML.png")
plt.show()

sns.scatterplot(x="SS", y="fC",hue = 'Clusters',  data=datos, palette='viridis')
plt.savefig("SS_fC.png")
plt.show()