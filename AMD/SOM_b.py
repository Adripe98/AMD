import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from minisom import MiniSom
import psycopg2

#importacion de datos
try:
    #parametros de conexion
    conn_str = "host='localhost' dbname='wdi' port=5432 user='postgres' password='postgres'"
    print("Conectando a la BD\n->%s" % (conn_str))
    #intento de conexion
    conn = psycopg2.connect(conn_str)
    print("DB conectada\n")
    #query e intento de traernos los datos a un dataframe
    SQL_query = "select paiscode, fertilidadadol, mortalidadinfantil, pibcapita, esperanzavida, rentacapita, electricidadcapita, co2capita, gastosanidadcapita from datos where anocode like 'YR2014'"
    datos_raw=pd.read_sql(SQL_query, con=conn)
except (Exception, psycopg2.Error) as error:
    #manejo de excepciones
    print("Error en la conexión PostgreSQL", error)
finally:
    #al final siempre cerramos la conexion si logramos abrirla antes
    if conn:
        conn.close()
        print("Conexión PostgreSQL cerrada")

#comprobamos si existen nulos o nan y la estructura de los datos
print(datos_raw.isnull().sum())
print(datos_raw.isna().sum())
datos_raw.info()

#elegir columnas; pasamos a array por necesidad de MiniSom
valores=datos_raw[['fertilidadadol','mortalidadinfantil','pibcapita', 'esperanzavida', 'rentacapita', 'electricidadcapita', 'co2capita', 'gastosanidadcapita']]
datos=valores.values

#definimos y entrenamos el modelo SOM
som_params=(1,5)
som=MiniSom(som_params[0], som_params[1], datos.shape[1], sigma=0.5, learning_rate=0.7, random_seed=1234)
som.train_batch(datos, 1000, verbose=True)

#visualizacion de resultados
ganadores=np.array([som.winner(x) for x in datos]).T
clusters=np.ravel_multi_index(ganadores, som_params)

plt.figure(figsize=(6,6))
for c in np.unique(clusters):
    plt.scatter(datos[clusters == c, 0],
                datos[clusters == c, 2], label='cluster'+str(c), alpha=.7)
    plt.legend()
    plt.title('Fertilidad de adolescentes frente a PIB per cápita')
    filen='fert_pib.png'
    plt.savefig(filen)
    
plt.figure(figsize=(6,6))
for c in np.unique(clusters):
    plt.scatter(datos[clusters == c, 1],
                datos[clusters == c, 2], label='cluster'+str(c), alpha=.7)
    plt.legend()
    plt.title('Mortalidad infantil frente a PIB per cápita')
    filen='mort_pib.png'
    plt.savefig(filen)
    
plt.figure(figsize=(6,6))
for c in np.unique(clusters):
    plt.scatter(datos[clusters == c, 1],
                datos[clusters == c, 3], label='cluster'+str(c), alpha=.7)
    plt.legend()
    plt.title('Mortalidad infantil frente a Esperanza de vida')
    filen='mort_expec.png'
    plt.savefig(filen)
    
plt.figure(figsize=(6,6))
for c in np.unique(clusters):
    plt.scatter(datos[clusters == c, 2],
                datos[clusters == c, 4], label='cluster'+str(c), alpha=.7)
    plt.legend()
    plt.title('PIB per cápita frente a Renta per cápita')
    filen='pib_ren.png'
    plt.savefig(filen)
    
plt.figure(figsize=(6,6))
for c in np.unique(clusters):
    plt.scatter(datos[clusters == c, 5],
                datos[clusters == c, 6], label='cluster'+str(c), alpha=.7)
    plt.legend()
    plt.title('Electricidad consumida per cápita frente a Emisiones de CO2 per cápita')
    filen='elec_co.png'
    plt.savefig(filen)
    
plt.figure(figsize=(6,6))
for c in np.unique(clusters):
    plt.scatter(datos[clusters == c, 1],
                datos[clusters == c, 7], label='cluster'+str(c), alpha=.7)
    plt.legend()
    plt.title('Mortalidad infantil frente a Gasto en sanidad per cápita')
    filen='mort_sani.png'
    plt.savefig(filen) 