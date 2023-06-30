import numpy as np
import pandas as pd
from apyori import apriori

# Importamos el dataset
dataset = pd.read_csv('cesta_compra2_1.csv', header = None)

# Extraemos los items de cada compra
compras = []
for i in range(0, 14964):
    a = dataset.values[i,2].split(";")
    compras.append([a[j] for j in range(len(a))])


# Aplicamos el algoritmo apriori
rules = apriori(compras, min_support = 0.003, min_confidence = 0.05, min_lift = 1, min_length = 3)

# Resultados
results = list(rules)

# Función para facilitar la visualización de los resultados
def inspecionar(results):
    rh          = [tuple(result[2][0][0]) for result in results]
    lh          = [tuple(result[2][0][1]) for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(rh, lh, supports, confidences, lifts))

# Este comando crea un frame para ver los datos resultados
resultDataFrame=pd.DataFrame(inspecionar(results),
                columns=['rhs','lhs','support','confidence','lift'])

#Imprimimos el frame con las reglas
print(resultDataFrame)