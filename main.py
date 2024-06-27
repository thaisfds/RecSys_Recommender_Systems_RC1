#imports
import sys
import pandas as pd
import numpy as np

''' Tratamento dos dados:
    - Ratings possui as interações usuário-item
    - Targets possui os dados para qual precisamos prever os resultados
'''
def manipulate_dataframe(ratings_file, targets_file):

    # Ler o arquivo CSV original para um DataFrame
    df_ratings_original = pd.read_csv(ratings_file, sep=',', header=0, names=["UserId:ItemId", "Rating"])

    # Dividir a coluna "UserId:ItemId" em duas colunas separadas usando ":"
    df_ratings_split = df_ratings_original["UserId:ItemId"].str.split(":", expand=True)
    df_ratings_split.columns = ["UserId", "ItemId"]

    # Concatenar as colunas divididas com a coluna "Rating" em um novo DataFrame
    df_ratings = pd.concat([df_ratings_split, df_ratings_original["Rating"]], axis=1)

    # Ler o arquivo CSV targets.csv para df_targets_original
    df_targets = pd.read_csv(targets_file, sep=':', header=0, names=["UserId","ItemId"])

    #Adicionar a coluna "Rating" ao DataFrame df_targets_original
    df_targets["Rating"] = 0

    return df_ratings, df_targets



#Para um usuário/item no dataframe targets, calcula o rating previsto usando a função SGD
#Arredonda os valores para 1 ou 5 caso o valor previsto seja menor que 1 ou maior que 5
def predict_rating(user, item, P, Q, user_index, item_index):
    if user in user_index and item in item_index:
        result = np.dot(P[user_index[user]], Q[item_index[item]].T)
        if result > 5:
            result = 5
        elif result < 1:
            result = 1
        return result

#Para cada usuário/item no dataframe targets, calcula o rating previsto usando a função SGD
def predict_all_ratings(df_targets, P, Q, user_index, item_index):
    df_targets["Rating"] = df_targets.apply(lambda x: predict_rating(x["UserId"], x["ItemId"], P, Q, user_index, item_index), axis=1)
    return df_targets

''' SGD (Stochastic gradient descent)
    - S é o dataframe que armazena os ratings de treino
    - k é o tamanho dos vetores latentes 
    - alpha é a taxa de aprendizado
    - lamb é a variável de regularização
    - epochs é o número de épocas usadas no algoritmo
'''

def SGD(S, k=18, alpha=0.01, lamb=0.02, epochs=10):

    #Contar o numero de usuarios e itens unicos
    n_users = S["UserId"].nunique()
    n_items = S["ItemId"].nunique()

    #Criar um dicionario para mapear os usuarios e itens
    user_index = {original: new for new, original in enumerate(S["UserId"].unique())}
    item_index = {original: new for new, original in enumerate(S["ItemId"].unique())}

    np.random.seed(15)

    # Inicializar os vetores de usuário e item com valores aleatórios
    P = np.random.rand(n_users, k)
    Q = np.random.rand(n_items, k)

    SList = S[["UserId", "ItemId", "Rating"]].values
    
    #Para cada periodo de treino
    for epoch in range(epochs):

        #erroraccum = 0 #Para calcular o erro médio quadrático descomente essa linha
        #count = 0 #Para calcular o erro médio quadrático descomente essa linha

        #Para cada usuário/item no dataframe de treino
        for user, item, rating in SList:
            i = user_index[user]
            j = item_index[item]

            #calcula o erro
            error = rating - np.dot(P[i, :], Q[j, :].T)

            #atualiza os valores de P e Q, Pnew e Qnew são utilizados para não alterar os dados atuais de P e Q antes de atualizar todos os valores
            for l in range(k):
                Pnew = error * Q[j,l]
                Qnew = error * P[i,l]
                P[i,l] += alpha * (Pnew - lamb * P[i,l])
                Q[j,l] += alpha * (Qnew - lamb * Q[j,l])
            
            #erroraccum += error**2 #Para calcular o erro médio quadrático descomente essa linha
            #count +=1 #Para calcular o erro médio quadrático descomente essa linha
        
        #mse = erroraccum / count #Para calcular o erro médio quadrático descomente essa linha
        #print("Epoch: " + str(epoch+1) + " Error: " + str(np.sqrt(mse))) #Para calcular o erro médio quadrático descomente essa linha

    
    return P, Q, user_index, item_index


#le os arquivos de entrada e chama as funções para manipular os dados
ratings_file = sys.argv[1]
targets_file = sys.argv[2]
df_ratings, df_targets = manipulate_dataframe(ratings_file, targets_file)

#Executa o algoritmo SGD para os ratings de treino
P, Q, user_index, item_index = SGD(df_ratings)

#Calcula os ratings previstos para os ratings de teste
df_targets = predict_all_ratings(df_targets, P, Q, user_index, item_index)

#Juntar UserId e ItemId e Rating em uma coluna para gerar o arquivo de submissão final da forma UserId:ItemId,Rating
df_targets["UserId:ItemId"] = df_targets["UserId"].astype(str) + ":" + df_targets["ItemId"].astype(str)
df_targets = df_targets[["UserId:ItemId", "Rating"]]

#imprimir df_targets na tela com UserId:ItemId,Rating na primeira linha e os valores nas linhas seguintes
print("UserId:ItemId,Rating")
for index, row in df_targets.iterrows():
    print(row["UserId:ItemId"] + "," + str(row["Rating"]))

