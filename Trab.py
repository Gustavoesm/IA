
from pandas import DataFrame, read_csv # for reading tsv files and structuring
import math # for mathematical functions
import numpy as np # for vector stuff
import matplotlib.pyplot as plt # for plotting graphs
from scipy.spatial import distance, distance_matrix # for euclidian distance
from random import randint # for generating random colors
import time
import sklearn.metrics as metrics

dataset1 = read_csv("datasets/c2ds1-2sp.txt", sep="\t", index_col="sample_label") # Lê o arquivo de dados
dataset2 = read_csv("datasets/c2ds3-2g.txt", sep="\t", index_col="sample_label") # Lê o arquivo de dados
dataset3 = read_csv("datasets/monkey.txt", sep="\t", index_col="sample_label") # Lê o arquivo de dados



def single_link(dataframe, k_min, k_max):
        
    start = time.time()

        
    # creates a distance matrix in dataframe format
    dist_matrix = DataFrame(distance_matrix(dataframe.values, dataframe.values))

    clusters = []
    resultados = []
    
    count = 0
    for sample in dataframe.iterrows():
        clusters.append([sample])
            
    for k in range(len(dataframe), k_min, -1):
        
        closest_idx = []
        closest_values = []

        # initializing position and possible closest
        pos = 0
        possible_closest = [0,1]
        
        # for each row
        while(pos < len(dist_matrix)-1):

            # find closest pair of this row
            possible_closest = [dist_matrix.index[pos], dist_matrix.iloc[pos, pos+1:].idxmin()]

            # add index and values to respectives possible's list
            closest_idx.append(possible_closest)
            closest_values.append(dist_matrix.at[possible_closest[0], possible_closest[1]])

            # iteration end
            pos+=1
            
        # find the closest pair of all clusters
        [p1, p2] = closest_idx[closest_values.index(min(closest_values))]

        # find minimum distances in p1 and p2, saves them in p1 
        for index in dist_matrix.columns:
            dist_matrix.at[p1, index] = dist_matrix.at[index, p1] = min(dist_matrix.at[p1, index], dist_matrix.at[p2, index])

        # get positional indexes for list structures
        ip1 = dist_matrix.index.get_loc(p1)
        ip2 = dist_matrix.index.get_loc(p2)
        
        # merge clusters (p1 and p2) 
        clusters[ip1] += clusters.pop(ip2)
                
         # drops p2 for it's irrelevance, p2 is inside p1 now
        dist_matrix = dist_matrix.drop(index=p2, columns=p2)

        if ( len(clusters) <= k_max):
            resultados.append(clusters.copy())
        
        # Optional completion feedback
#         print("Iteration complete, %d clusters remaining" % (k-1))
    end = time.time()
    return [resultados, end-start]


k_min = 5
k = k_max = 12

with open('datasets/c2ds1-2spReal.clu', 'r') as f:
    realResult = [[entry for entry in line.split()] for line in f.readlines()]
realResult = DataFrame(realResult)
#print(realResult[1])

resultados = single_link(dataset1, 2, k_max)


print("Tempo total de execução: %.2f segundos" % resultados[1])
for resultado in resultados[0]:
    manipulated_df = dataset1.copy() # setting up manipulated dataset for use
    manipulated_df['cluster_id'] = None # adding a column for cluster

    cluster_id = 0
    for cluster in resultado:
        for index, sample in cluster:
            manipulated_df.at[index, 'cluster_id'] = cluster_id
        cluster_id += 1

    manipulated_df.plot.scatter('d1', 'd2', c='cluster_id', colormap='gist_rainbow')
    plt.title("Single-Link for dataset 3 (monkey) with k = %d" % k)
    plt.show()
    k -= 1

    print("ARI = " + str(metrics.adjusted_rand_score(manipulated_df['cluster_id'], realResult[1])))