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
import psycopg2

try:
    conn_str = "host='localhost' dbname='p7' port=5432 user='postgres' password='a'"
    print("Conectando a la BD\n->%s" % (conn_str))
    # Conectamos y si no se puede realizar, lanzamos una excepción
    conn = psycopg2.connect(conn_str)
    # conn.cursor devuelve un objeto cursor que usaremos para realizar las consultas
    cursor = conn.cursor()
    print("DB conectada\n")

    SQL_query = "select adolescent_fertility_rate, life_expectancy_at_birth from data"

    cursor.execute(SQL_query)

    dt = cursor.fetchall()

    datos = pd.DataFrame(dt, columns=['adolescent_fertility_rate','life_expectancy_at_birth'])

    print("Dataset:",datos.shape)

    scaler = MinMaxScaler()
    scale = scaler.fit_transform(datos[['adolescent_fertility_rate','life_expectancy_at_birth']])
    datos_scale = pd.DataFrame(scale, columns = ['adolescent_fertility_rate','life_expectancy_at_birth']);
    #print(datos_scale.head(5))

    km=KMeans(n_clusters=2, n_init=10)
    y_predicted = km.fit_predict(datos[['adolescent_fertility_rate','life_expectancy_at_birth']])
    #print(y_predicted)

    print(km.cluster_centers_)

    datos['Clusters'] = km.labels_
    sns.scatterplot(x='life_expectancy_at_birth', y='adolescent_fertility_rate',hue = 'Clusters',  data=datos,palette='viridis')
    #plt.show()


    # Encontrar el número de clusters óptimo
    for i in range(2,6):
        labels = cluster.KMeans(n_clusters=i,random_state=200, n_init=10).fit(datos_scale).labels_
        print ("Silhouette score for k(clusters) = "+str(i)+" is "
        +str(metrics.silhouette_score(datos_scale,labels,metric="euclidean",sample_size=1000,random_state=200)))
        #El valor óptimo de k es 2

    kmeans = cluster.KMeans(n_clusters=2 ,init="k-means++")
    kmeans = kmeans.fit(datos[['adolescent_fertility_rate','life_expectancy_at_birth']])

    datos['Clusters'] = kmeans.labels_

    sns.scatterplot(x='life_expectancy_at_birth', y='adolescent_fertility_rate',hue = 'Clusters',  data=datos, palette='viridis')
    plt.savefig("DB.png")
    plt.show()

except (Exception, psycopg2.Error) as error:
    print("Error en la conexión PostgreSQL", error)

finally:
    # closing database connection.
    if conn:
        cursor.close()
        conn.close()
        print("Conexión PostgreSQL cerrada")