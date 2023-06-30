#Problema de regresión con Random Forest

#Importaciones necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

#definimos la semilla aleatoria para hacer los resultados consistentes e importamos el dataset
SEED_VALUE=123456789
dataset=pd.read_csv('AMD_regresion.csv')

#partimos el dataset en independientes y dependientes, y realizamos un split en conjunto de entrenamiento y test
#con proporcion 70/30 (bastante estándar)
indep=dataset[['LA','CL','SL','JD','ST','SS','CH','ML','U2']]
dep=dataset[['fC']]
indep_train, indep_test, dep_train, dep_test=train_test_split(indep, dep, random_state=SEED_VALUE, train_size=0.7)

#creacion del modelo y predicciones
regresor=RandomForestRegressor(n_estimators=100, random_state=SEED_VALUE)
regresor.fit(indep_train, np.ravel(dep_train))

pred_test=regresor.predict(indep_test)

#transformamos los valores predichos y los reales a una Serie de pandas para poder calcular el 
#coeficiente de correlacion además del error cuadrático medio
test_s=pd.Series(np.ravel(dep_test.values))
pred_s=pd.Series(pred_test)
correlacion=test_s.corr(pred_s)

#métricas de precisión
print("Raíz del error cuadrático medio: ", metrics.mean_squared_error(dep_test,pred_test,squared=False))
print("Coeficiente de correlación entre los valores predichos y los reales: ", correlacion)