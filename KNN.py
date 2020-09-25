import numpy as np
import operator
import matplotlib.pyplot as plt


def euclidean_distance(ind1, ind2):
    distance = 0.0
    for i in range(132):
        # print(ind1[i])
        distance += (ind1[i] - ind2[i])**2

    # print(distance)
    return np.sqrt(distance)

def manhattan_distance(ind1,ind2):
    distance = 0.0


def get_classe_vizinhos(aux, vizinhos, resultado):
    indexes = []
    for i in range(len(aux)):
        for j in range(len(vizinhos)):
            if aux[i] == vizinhos[j]:
                indexes.append(i)
    # print (indexes)
    
    retorno_classe=[]
    for i in indexes:
        retorno_classe.append(resultado[i])

    # print(retorno_classe)   
    return retorno_classe

def get_Neighbors(treino, ind_teste, k):
    distancia = []
    resultado = []
    for i in range(1000):
        distancia.append([])
        for j in range(133):
            if (j == 132):
                # print ( teste[i][j])
                resultado.append(treino[i][j])
        distancia[i].append(euclidean_distance(treino[i], ind_teste))
    aux = distancia.copy()
    distancia.sort()
    vizinhos = []
    for i in range(k):
        vizinhos.append(distancia[i][0])
    # print (vizinhos)
    get_classe_vizinhos(aux,vizinhos,resultado)
    return aux, vizinhos, resultado
        
    # print (resultado)
    # print(distancia)

def predict_classification(treino, ind_teste, k):
    aux, vizinhos, resultado = get_Neighbors(treino, ind_teste, k)
    vizinhos.reverse()
    # print(vizinhos)
    for i in vizinhos:
        output_values = i
    # print (output_values)
    prediction = get_classe_vizinhos(aux, [output_values], resultado)
    prediction = prediction[0]
    return prediction

def knn(treino, teste, k):
    predictions = []
    counter = 0
    resultado = []
    for i in teste:
        resultado.append(i[-1])
        output = predict_classification(treino, i, k)
        predictions.append(output)
        print('Individuo classificado: %d,\n Esperado %d,\n Adquirido %d.' % (counter, i[-1], output))
        counter+=1
    return(predictions)

teste = np.loadtxt("teste.txt",
               delimiter=' ')

treinamento = np.loadtxt("treinamento.txt",
                delimiter=' ')

# prediction = predict_classification(f2, f[100], 3)
k = [1,3,5,7,9,11,13,15,17,19]
# for i in range(10):
    # print (k[i])
knn = knn(treinamento, teste, 9)
