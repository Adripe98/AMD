#Problema de clasificación

#Se va a utilizar primero Random Forest, por ser la mejor familia de clasificadores según un artículo científico leído (citado en
#memoria); y después un Decision Tree por ser este más intuitivo y sencillo de visualizar. Además, así podremos comparar sus
#desempeños

#importaciones necesarias para todo el código
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pydot
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn import metrics
from io import StringIO
from sklearn.metrics import ConfusionMatrixDisplay

#ponemos una semilla aleatoria para hacer los resultados consistentes e importamos los datasets
SEED_VALUE=123456789
test=pd.read_csv('clasificar_test.csv')
train=pd.read_csv('clasificar_train.csv')

#separacion de variables en independientes y dependientes
#train
indep_train=train[['landsat__1','landsat__2','landsat__3','landsat__4','landsat__5','landsat__6','landsat__7']]
dep_train=train[['Cat']]
#test
indep_test=test[['landsat__1','landsat__2','landsat__3','landsat__4','landsat__5','landsat__6','landsat__7']]
dep_test=test[['Cat']]

#RANDOM FOREST
clasificador=RandomForestClassifier(n_estimators=100, random_state=SEED_VALUE)
clasificador.fit(indep_train, np.ravel(dep_train))
pred_test=clasificador.predict(indep_test)

#Evaluamos la precisión del modelo y vemos predicciones y valores reales.
print("Accuracy: ",metrics.accuracy_score(dep_test, pred_test))

print(pred_test)

print(np.ravel(dep_test))

#generación de matrices de confusión para el conjunto de train y test
disp1=ConfusionMatrixDisplay.from_estimator(clasificador, indep_train, dep_train,
                      display_labels=clasificador.classes_,
                      cmap=plt.cm.Blues,xticks_rotation=25)

disp1.ax_.set_title("Entrenamiento-RF")


disp2=ConfusionMatrixDisplay.from_estimator(clasificador, indep_test, dep_test,
                      display_labels=clasificador.classes_,
                      cmap=plt.cm.Blues,xticks_rotation=25)

disp2.ax_.set_title("Test-RF")



#CART (Classification and Regression Trees)
clasificador=DecisionTreeClassifier(min_impurity_decrease=0.006, random_state=SEED_VALUE)
clasificador.fit(indep_train, np.ravel(dep_train))
pred_test=clasificador.predict(indep_test)

#mismo calculo de precisión
print("Accuracy: ",metrics.accuracy_score(dep_test, pred_test))

print(pred_test)

print(np.ravel(dep_test))

#visualizamos el árbol (antes no lo hacíamos porque el Random Forest trabaja con 100 árboles a la vez)
dot_data = StringIO()

export_graphviz(clasificador, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,
                feature_names=indep_train.columns,
                leaves_parallel=True,
                label='none',
                class_names=clasificador.classes_,
                impurity=False,
                node_ids=True,
                precision=0)

graph = pydot.graph_from_dot_data(dot_data.getvalue())[0]
graph.write_png('CART_display.png')

#matrices de confusión
disp3=ConfusionMatrixDisplay.from_estimator(clasificador, indep_train, dep_train,
                      display_labels=clasificador.classes_,
                      cmap=plt.cm.Blues,xticks_rotation=25)

disp3.ax_.set_title("Entrenamiento-CART")

disp4=ConfusionMatrixDisplay.from_estimator(clasificador, indep_test, dep_test,
                      display_labels=clasificador.classes_,
                      cmap=plt.cm.Blues,xticks_rotation=25)

disp4.ax_.set_title("Test-CART")

plt.show()
