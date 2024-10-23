import numpy as np
import pandas as pd
from scipy.spatial import distance
import matplotlib.pyplot as plt
import seaborn as sn

library = pd.read_csv('dataset_doencas.csv')  #dataset com os casos originais passados
cases = pd.read_csv('dataset_casos.csv')  #dataset com os casos a serem diagnosticados

cases['Outcome Variable'] = None  

for i in range(cases.shape[0]):
    case_row = cases.loc[i, :].copy()  
    disease = case_row['Disease']  #pega a doença que ta analisando atualmente

    #pega só as doenças que são iguais 
    relevant_cases = library[library['Disease'] == disease]

    if relevant_cases.empty:
        print(f'> For case/problem {i}: {case_row.to_numpy()}, no relevant cases found.')
        continue

    covariance_matrix = relevant_cases.iloc[:, :-1].cov()
    inverse_covariance_matrix = np.linalg.pinv(covariance_matrix)

    #guarda distancia mahalanobis
    distances = np.zeros(relevant_cases.shape[0])

    for j in range(relevant_cases.shape[0]):
        base_row = relevant_cases.iloc[j, :-1].to_numpy() 
        
        #exclue a doença do calculo da distancia
        weighted_case_row = case_row[:-1].to_numpy() 

        #calcula a distancia
        distances[j] = distance.mahalanobis(weighted_case_row, base_row, inverse_covariance_matrix)

    #pega a distancia de menor valor
    min_distance_row = np.argmin(distances)

    #solução vira a menor distancia
    case = np.append(cases.iloc[i, :-1].to_numpy(), relevant_cases.iloc[min_distance_row, -1:])  

    case_series = pd.Series(case[:len(library.columns)], index=library.columns) 
    cases.loc[i, 'Outcome Variable'] = relevant_cases.iloc[min_distance_row, -1]  

    library = pd.concat([library, case_series.to_frame().T], ignore_index=True) #adiciona os novos diagnosticos à base original

library.to_csv('library.csv', index=False)  #salva o dataset atualizado com os diagnosticos
cases.to_csv('diagnostico.csv', index=False)  #salva os diagnosticos sozinhos
