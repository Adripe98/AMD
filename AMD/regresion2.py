#Problema de regresión con redes neuronales (MLP, Multi Layer Perceptron)

#importaciones necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
from sklearn.model_selection import train_test_split

#definimos semilla aleatoria para hacer consistentes los resultados e importamos el dataset
SEED_VALUE=123456789
dataset=pd.read_csv('AMD_regresion.csv')

#separacion de variables en dependientes e independientes, y generación de los conjuntos de entramiento y test
#con proporción 70/30
indep=dataset[['LA','CL','SL','JD','ST','SS','CH','ML','U2']]
dep=dataset[['fC']]
indep_train, indep_test, dep_train, dep_test=train_test_split(indep, dep, random_state=SEED_VALUE, train_size=0.7)

#definimos el modelo de regresor, con 10 capas ocultas, función sigmoide para la activación, y máximo de 10.000 iteraciones
#despues ajustamos y predecimos
regresor=MLPRegressor(hidden_layer_sizes=10, max_iter=10000, activation='logistic', random_state=SEED_VALUE)
regresor.fit(indep_train, np.ravel(dep_train))

pred_test=regresor.predict(indep_test)

#calculamos los mismos parametros de ajuste que con Random Forest
test_s=pd.Series(np.ravel(dep_test.values))
pred_s=pd.Series(pred_test)
correlacion=test_s.corr(pred_s)

print("Raíz del error cuadrático medio: ", metrics.mean_squared_error(dep_test,pred_test,squared=False))
print("Coeficiente de correlación entre los valores predichos y los reales: ", correlacion)


#Tras las pruebas, he visto que lo anterior no ajusta demasiado bien, así que voy a probar a normalizar
#las entradas (en este caso min-max), ya que en redes neuronales suele ser conveniente hacerlo.

#partimos en train y test antes de normalizar, ya que si no lo hacemos se va a "contaminar" los datos: ambos
#conjuntos deben normalizarse en la misma escala, para no introducir sesgos, pero esa escala debe definirse
#únicamente sobre el conjunto de train, ya que si tomamos los valores de test en consideración, en cierto
#modo estamos dando a la red información sobre ese conjunto, que nunca debería tener.
train, test=train_test_split(dataset, random_state=SEED_VALUE, train_size=0.7)

#min-max
maximo=train.max()
minimo=train.min()
train_n=(train - minimo) / (maximo - minimo)
test_n=(test - minimo) / (maximo - minimo)

#separación de variables
indep_train=train_n[['LA','CL','SL','JD','ST','SS','CH','ML','U2']]
dep_train=train_n[['fC']]
indep_test=test_n[['LA','CL','SL','JD','ST','SS','CH','ML','U2']]
dep_test=test_n[['fC']]

#definimos el mismo modelo que antes, y predecimos
regresor=MLPRegressor(hidden_layer_sizes=10, max_iter=10000, activation='logistic', random_state=SEED_VALUE)
regresor.fit(indep_train, np.ravel(dep_train))

pred_test=regresor.predict(indep_test)

#calculamos parametros de ajuste
test_s=pd.Series(np.ravel(dep_test.values))
pred_s=pd.Series(pred_test)
correlacion=test_s.corr(pred_s)

print("Raíz del error cuadrático medio: ", metrics.mean_squared_error(dep_test,pred_test,squared=False))
print("Coeficiente de correlación entre los valores predichos y los reales: ", correlacion)



#Tras las pruebas, el cambio es poco apreciable. Vamos a continuar con valores normalizados, pero tocar un poco el modelo
#de red, para intentar tener un ajuste mejor.

#regresor de 7 capas ocultas, 1000 iteraciones máximas (cualquier número superior tarda demasiado), función de activación sigmoide,
#un learning rate inicial de 0.0035 y sin límite de iteraciones "estancado" (véase, que nunca pare antes de las iteraciones máximas).
regresor=MLPRegressor(hidden_layer_sizes=7, max_iter=1000, activation='logistic', solver='adam',
                                learning_rate_init=0.0035, n_iter_no_change=float('inf'), random_state=SEED_VALUE)
regresor.fit(indep_train, np.ravel(dep_train))

pred_test=regresor.predict(indep_test)

#calculamos parametros de ajuste
test_s=pd.Series(np.ravel(dep_test.values))
pred_s=pd.Series(pred_test)
correlacion=test_s.corr(pred_s)

print("Raíz del error cuadrático medio: ", metrics.mean_squared_error(dep_test,pred_test,squared=False))
print("Coeficiente de correlación entre los valores predichos y los reales: ", correlacion)