import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from minisom import MiniSom

#leer csv completo
datos_raw=pd.read_csv('AMD_clustering_0.csv')

#sacamos un 5% de los datos por cuestiones de rendimiento
datos_sub=datos_raw.sample(frac=0.05)

#comprobamos si existen nulos o nan
print(datos_sub.isnull().sum())
print(datos_sub.isna().sum())

#elegir columnas; pasamos a array por necesidad de MiniSom; y hacemos una lista para ayudarnos
#con los plots
valores=datos_sub[['SS','ST','CH','ML','fC']]
datos=valores.values
plot_list=[0,1,2,3,4]

#definimos y entrenamos el modelo SOM
som_params=(1,5)
som=MiniSom(som_params[0], som_params[1], datos.shape[1], sigma=0.5, learning_rate=0.7, random_seed=1234)
som.train_batch(datos, 1000, verbose=True)

#visualizacion de resultados
ganadores=np.array([som.winner(x) for x in datos]).T
clusters=np.ravel_multi_index(ganadores, som_params)

for i in range(5):
    this_plot_list=[val for val in plot_list if val!=i]
    for j in this_plot_list:
        plt.figure(figsize=(6,6))
        for c in np.unique(clusters):
            plt.scatter(datos[clusters == c, i],
                        datos[clusters == c, j], label='cluster'+str(c), alpha=.7)
            plt.legend()
            plt.title(datos_sub.columns[i]+' frente a '+datos_sub.columns[j])
            filen=datos_sub.columns[i]+'_'+datos_sub.columns[j]+'.png'
            plt.savefig(filen)           