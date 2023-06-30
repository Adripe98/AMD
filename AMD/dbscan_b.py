import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import psycopg2

data = pd.read_csv('AMD_clustering_0.csv',  # Cargamos el csv
                       delimiter=',',
                       header=0)


try:
    conn_str = "host='localhost' dbname='p7' port=5432 user='postgres' password='a'"
    print("Conectando a la BD\n->%s" % (conn_str))
    # Conectamos y si no se puede realizar, lanzamos una excepción
    conn = psycopg2.connect(conn_str)
    # conn.cursor devuelve un objeto cursor que usaremos para realizar las consultas
    cursor = conn.cursor()
    print("DB conectada\n")

    SQL_query = "select adolescent_fertility_rate, life_expectancy_at_birth, mortality_rate, gdp_per_capita from data"

    cursor.execute(SQL_query)

    dt = cursor.fetchall()

    data = pd.DataFrame(dt, columns=['adolescent_fertility_rate','life_expectancy_at_birth','mortality_rate', 'gdp_per_capita'])

    data.head();
    print("Dataset shape:", data.shape);
    data.isnull().any().any();

    modelo=DBSCAN(eps=0.0000005, min_samples=1)
    clusterDBSCAN=data.copy()
    clusterDBSCAN['dbscan']=modelo.fit_predict(clusterDBSCAN.to_numpy())
    sb.pairplot(clusterDBSCAN, hue='dbscan')
    plt.show()

except (Exception, psycopg2.Error) as error:
        print("Error en la conexión PostgreSQL", error)

finally:
    # closing database connection.
    if conn:
        cursor.close()
        conn.close()
        print("Conexión PostgreSQL cerrada")