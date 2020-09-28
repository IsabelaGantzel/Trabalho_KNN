import numpy as np
import operator
import random
from sklearn.metrics import confusion_matrix as cm
import scipy.stats as stats

# Encontra o minmax treino e do teste
def minmax(dataset):
    minmax = []
    x = np.array(dataset)
    print ("teste: ", x)

    value_min = min(x)
    value_max = max(x)
    minmax.append([value_min, value_max])

    # for i in range(len(x)):
    #     col_values = x[i]
    #     value_min = min(col_values)
    #     value_max = max(col_values)
    #     minmax.append([value_min, value_max])
    return minmax
    
# Normaliza usando zscore para treino e teste com: Z = (X - µ) / σ
def zscore(dataset):
    X = dataset
    print("vc ", stats.zscore(X, axis = 1))
    mean = np.mean(X) 
    std = np.std(X)
    # print(std)
    z_list = []

    for i in range (len(X)):
        z_list.append([])
        for j in range(len(X[i])):
            z = (X[i][j]-mean)/std
            z_list[i].append(z)
    print("eu ", np.array(z_list))
    return (z_list)
 
# Normaliza usando minmax
def normalizar(dataset):
    X = dataset[:,:-1]
    #print (X.shape)
    min = X.min()
    max = X.max()
    # print (min)
    # print (max)
    normalized_list = []
    for i in range (len(X)):
        normalized_list.append([])
        for j in range(len(X[i])):
            X_std = (float(X[i][j] - min)) / (float(max - min))
            X_scaled = (X_std * (10 - 1) + 1.0)
            normalized_list[i].append(X_scaled)
    #print ((normalized_list))
    return normalized_list

# Calcula a distância euclidiana
def euclidean_distance(ind1, ind2):
    distance = 0.0
    for i in range(132):
        # print(ind1[i])
        distance += (ind1[i] - ind2[i])**2
    # print(distance)
    return np.sqrt(distance)

#totaldistancesum faz a distancia de manhattan
def totaldistancesum(ind1, ind2): 
    sum = 0
    for i in range(len (ind1)):
        sum += abs(ind1[i] - ind2[i])
    #print (sum)
    return sum 

# Pega a classe dos vizinhos e adiciona-os em uma lista
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
    #print(retorno_classe)   
    return retorno_classe

#Pega k vizinhos do individuo teste
def get_Neighbors(treino, ind_teste, k):
    distancia = []
    resultado = []
    for i in range(len(treino)):
        distancia.append([])
        for j in range(133):
            if (j == 132):
                #print ( teste[i][j])
                resultado.append(treino[i][j])
        distancia[i].append(euclidean_distance(treino[i], ind_teste))
        #distancia[i].append(totaldistancesum(treino[i],ind_teste))
    aux = distancia.copy()
    distancia.sort()
    vizinhos = []
    for i in range(k):
        vizinhos.append(distancia[i][0])
    # print (vizinhos)
    #get_classe_vizinhos(aux,vizinhos,resultado)
    # print (resultado)
    #print(distancia)
    return aux, vizinhos, resultado     

#Faz uma predição para ver quais sao os k vizinhos e atraves do max percebe-se qual é o adquirido
def predict_classification(treino, ind_teste, k):
    aux, vizinhos, resultado = get_Neighbors(treino, ind_teste, k)
    vizinhos.reverse()
    # print(vizinhos)
    # for i in vizinhos:
    #     output_values = i
    # # print (output_values)
    prediction = get_classe_vizinhos(aux, vizinhos, resultado)
    #print((prediction))
    prediction = max(set(prediction), key = prediction.count)
    #print(prediction)
    return prediction

#Faz a matriz confusão se o resultado for uma matriz identidade com valores 100 a precisão esta 100%
def get_confusion_matrix(r_obtained, r_expected):
    confusion_matrix = cm (r_obtained, r_expected)
    print(confusion_matrix)
    return confusion_matrix

#Calcula a precisão do knn atraves da comparação com o obtido e o esperado
def get_accuracy(r_obtido, r_esperado):
    correct = 0
    for i in range(len(r_obtido)):
        if r_obtido[i] == r_esperado[i]:
        	correct += 1

    get_confusion_matrix(r_obtido, r_esperado)
    return correct / float(len(r_obtido)) * 100.0

#Algoritmo KNN
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
    # print(predictions)
    #print(len(predictions))
    accuracy = get_accuracy(resultado, predictions)
    print("Precisão: ", accuracy)
    return(predictions)

#Separa o conjunto de treinamento (aleatoriamente) em x porcentagem dos dados de treinamento.
def def_treino (treinamento):
    tamanho = float(input("Digite um valor de porcentagem de treinamento: "))/100
    treino_index = (random.sample(range(1000), int(len(treinamento)*tamanho)))
    treino_final = []
    for i in treino_index:
        treino_final.append(treinamento[i])
    return treino_final

teste = np.loadtxt("teste.txt",
               delimiter=' ')

treinamento = np.loadtxt("treinamento.txt",
                delimiter=' ')

# prediction = predict_classification(f2, f[100], 3)
#Dentro do for avalia o desempenho para diferentes valores de k {1,3,5,7,9,11,13,15,17,19}
k = [1,3,5,7,9,11,13,15,17,19]
# for i in range(10):
    # print (k[i])
    #knn = knn(def_treino(treinamento), teste, k[i])
zscore(teste)
zscore(treinamento)

#normalizar(teste)
#normalizar(treinamento)
knn = knn(def_treino(treinamento), teste, 5)