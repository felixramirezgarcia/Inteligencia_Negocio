# -*- coding: utf-8 -*-
"""
Autor:
    Félix Ramírez García.
Fecha:
    10 de Noviembre de 2018.
Contenido:
    Uso de algoritmos de Clustering con diferentes librerías.
    Asignatura Inteligencia de Negocio.
    4º curso del Grado en Ingeniería Informática.
    Universidad de Granada.
"""
#--------------------------------------------------------------------------
#                           Imports
#--------------------------------------------------------------------------
#Realizamos los imports necesarios para la realizacion de la practica
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn.cluster import KMeans, AgglomerativeClustering,estimate_bandwidth
from sklearn.cluster import Birch,SpectralClustering,MeanShift
from sklearn import metrics, preprocessing
from math import floor
import seaborn as sns
import time
from scipy.cluster.hierarchy import dendrogram,ward,linkage
import numpy as np
from scipy.cluster import hierarchy
#--------------------------------------------------------------------------
#                           Funciones
#--------------------------------------------------------------------------
def makeScatterMatrix(data,outputName=None,displayOutput=True):
    sns.set()
    variables = list(data)
    variables.remove('cluster')
    sns_plot = sns.pairplot(data, vars=variables, hue="cluster", palette='Paired', plot_kws={"s": 25})
    sns_plot.fig.subplots_adjust(wspace=.03, hspace=.03);

    if displayOutput:
        plt.show()

    if outputName != None:
        outputName += ".png"
        print(outputName)
        plt.savefig(outputName)
        plt.clf()
#--------------------------------------------------------------------------  
def norm_to_zero_one(df):
    return (df - df.min()) * 1.0 / (df.max() - df.min())
#--------------------------------------------------------------------------  

def createLatexDataFrame(data):
    my_index = list(dict(data.items()).keys())
    my_data = list(data.values())
    my_cols = list(my_data[0].keys())
    latexDF = pd.DataFrame()

    for row in range(len(my_index)):
        aux = pd.DataFrame(data=my_data[row],index=[my_index[row]],columns=my_cols)
        latexDF = pd.concat([latexDF,aux])

    return latexDF
#--------------------------------------------------------------------------
def calculateMetrics(clusterPredict,data):
    metric_CH = metrics.calinski_harabaz_score(data, clusterPredict)
    metric_SC = metrics.silhouette_score(data, clusterPredict, metric='euclidean', sample_size=floor(0.1 * len(data)),
                                        random_state=123456)

    return [metric_CH,metric_SC]
#--------------------------------------------------------------------------
def createDataFrame(dataframe,prediction):
    cluster = pd.DataFrame(prediction,index=dataframe.index,columns=['cluster'])
    clusterX = pd.concat([dataframe,cluster],axis=1)

    return clusterX
#--------------------------------------------------------------------------
def createPrediction(dataframe,data,model):
    time_start = time.time()
    cluster_predict = model.fit_predict(data)
    time_finish = time.time() - time_start

    X_dataFrame = createDataFrame(dataframe,cluster_predict)

    return [calculateMetrics(cluster_predict,data),X_dataFrame,time_finish,cluster_predict]
#--------------------------------------------------------------------------
def calculateMeanDictionary(cluster,cluster_col = 'cluster'):
    vars = list(cluster)
    vars.remove(cluster_col)
    return dict(np.mean(cluster[vars],axis=0))
#--------------------------------------------------------------------------
def calculateDeviationDictionary(cluster, cluster_col = 'cluster'):
    vars = list(cluster)
    vars.remove(cluster_col)
    return dict(np.std(cluster[vars],axis=0))
#--------------------------------------------------------------------------
def createMeanClusterDF(dataFrame, clusterCol = 'cluster', cluster_print = 0):
    n_clusters = list(set(dataFrame[clusterCol]))

    my_mean_df = pd.DataFrame()
    my_deviation_df = pd.DataFrame()

    for cluster_n in n_clusters:
        my_cluster = dataFrame[dataFrame[clusterCol] == cluster_n]
        meanDic = calculateMeanDictionary(cluster=my_cluster,cluster_col = clusterCol)
        deviationDic = calculateDeviationDictionary(cluster=my_cluster, cluster_col = clusterCol)
        stdDF = pd.DataFrame(deviationDic, index=[str(cluster_n)])
        auxDF = pd.DataFrame(meanDic,index=[str(cluster_n)])
        my_mean_df = pd.concat([my_mean_df,auxDF])
        my_deviation_df = pd.concat([my_deviation_df,stdDF])

    if (cluster_print == 1):
        print(clusterCol)
        print(my_mean_df)

    return [my_mean_df, my_deviation_df]
#--------------------------------------------------------------------------
def findIndex(iterable, value):
    index = -1
    for val in iterable:
        if(value == val):
            return index

    return index
#--------------------------------------------------------------------------
def createNormalizedDF(dataFrame):
    vars = list(dataFrame)
    if(findIndex(vars,'cluster') != -1):
        vars.remove('cluster')

    norm = preprocessing.normalize(dataFrame,norm='l2')
    df = pd.DataFrame(norm,columns=vars, index=dataFrame.index)

    return df
#--------------------------------------------------------------------------
def makeHeatmap(data,displayOutput=True,outputName=None):
    meanDF, stdDF = createMeanClusterDF(dataFrame=data,cluster_print=1)
    meanDF = createNormalizedDF(dataFrame=meanDF)
    anotations = True
    sns.heatmap(data=meanDF, linewidths=.1, cmap='Blues_r', annot=anotations, xticklabels='auto')
    plt.xticks(rotation=0)

    if displayOutput:
        plt.show()

    if outputName != None:
        outputName += '.png'
        print(outputName)
        plt.savefig(outputName)
        plt.clf()
#--------------------------------------------------------------------------
def makeDendograma(data, displayOutput=True,outputName=None):
    meanDF, stdDF = createMeanClusterDF(dataFrame=data)   
    linkage_array = hierarchy.ward(meanDF)
    plt.figure()
    plt.clf()
    hierarchy.dendrogram(linkage_array)
    
    if displayOutput:
        plt.show()

    if outputName != None:
        outputName += '.png'
        print(outputName)
        plt.savefig(outputName)
        plt.clf()
#--------------------------------------------------------------------------        
def makeHeatMapConDendograma(data, displayOutput=True,outputName=None):
    meanDF, stdDF = createMeanClusterDF(dataFrame=data)   
    linkage_array = hierarchy.ward(meanDF)
    plt.figure()
    plt.clf()
    hierarchy.dendrogram(linkage_array ,orientation='left')
    sns.clustermap(meanDF, method='ward', col_cluster=False, figsize=(20, 10), linewidths=0.5, cmap='YlGn')
    
    if displayOutput:
        plt.show()

    if outputName != None:
        outputName += '.png'
        print(outputName)
        plt.savefig(outputName)
        plt.clf()
#--------------------------------------------------------------------------
#                           Programa principal
#--------------------------------------------------------------------------
    
carpeta_datos="C:/Users/felix/Desktop/Google_Drive/IN/Practica2/dataset/"

personas = pd.read_csv(carpeta_datos+'censo_granada.csv');
personas = personas.replace(np.NaN,0)

#--------------------------------------------------------------------------
#                           Casos de uso
#--------------------------------------------------------------------------

print("Caso de uso 3: Personas solteras mayores de 40 anios sin hijos en el hogar")

#                           Casos de uso 1

#subset = personas.loc[
#        (personas.H6584 >= 1) | (personas.H85M >= 1)
#        ]

#                           Casos de uso 2

#subset = personas.loc[
#        (personas.H6584 == 0) & (personas.H85M == 0) & (personas.HM5 == 0) & (personas.ECIVIL == 1)
#        ]

#                           Casos de uso 3

subset = personas.loc[
        (personas['ECIVIL'] == 1) & (personas['EDAD'] >= 40) & (personas['NHIJO'] == 0)
        ]

caso1 = ['TESTUD', 'TAMNUC', 'ESREAL','EDADPAD' ,'EDADMAD']

X = subset[caso1]

print("Tamaño del subconjunto de datos")

print(len(X))

#Indicamos la semilla al conjunto actual de dataset.
X = X.sample(len(X), random_state=123456)

#X_normal = preprocessing.normalize(X, norm='l2')
X_normal = X.apply(norm_to_zero_one)

k_means = KMeans(init='k-means++', n_clusters=10, n_init=5)
ward = AgglomerativeClustering(n_clusters=15,linkage='ward')
birch = Birch(n_clusters=10,threshold=0.1)
spectral = SpectralClustering(n_clusters=5,eigen_solver="arpack")
bandwidth = estimate_bandwidth(X_normal, quantile=0.4, random_state=123456)
meanshift = MeanShift(bandwidth=bandwidth,bin_seeding=True)

clustering_algorithms = (("K-means", k_means),
                         ("Birch", birch),
                         ("Ward", ward),
                         ("MeanShift",meanshift),
                         ("Spectral", spectral)
)

# Creamos los datos de salida, y mostramos las gráficas si queremos.
outputData = dict()
min_size = 5
t_global_start = time.time()

for algorithm_name,algorithm in clustering_algorithms:
    results = dict()
    met, clusterFrame, timeAlg,cluster_predict = createPrediction(dataframe=X, data=X_normal, model=algorithm)
    n_clusters=len(set(cluster_predict))

    if( n_clusters > 15 ):
        X_filtrado = clusterFrame[clusterFrame.groupby('cluster').cluster.transform(len) > min_size]
    else:
        X_filtrado = clusterFrame

    makeScatterMatrix(data=X_filtrado,outputName="./imagenes/scatterMatrix_caso3_" +algorithm_name,displayOutput=False)
    makeHeatmap(data=X_filtrado,outputName="./imagenes/heatmap_caso3_"+algorithm_name,displayOutput=False)
    makeDendograma(data=X_filtrado,outputName="./imagenes/dendograma_caso3_"+algorithm_name,displayOutput=False)
    makeHeatMapConDendograma(data=X_filtrado,outputName="./imagenes/heatmapcondendograma_caso3_"+algorithm_name,displayOutput=False)
    
    results['N Clusters']=n_clusters
    results['HC metric']=met[0]
    results['SC metric']=met[1]
    results['Time']=timeAlg

    outputData[algorithm_name] = results
    
t_global_final = time.time()

latexCaso3 = createLatexDataFrame(data=outputData)

f = open('caso3.txt','w')
f.write(latexCaso3.to_latex())
f.close()

print('\n{0:<15}\t{1:<10}\t{2:<10}\t{3:<10}\t{4:<10}'.format(
    'Name', 'N clusters', 'HC metric', 'SC metric', 'Time(s)'))

for name, res in outputData.items():
    print('{0:<15}\t{1:<10}\t{2:<10.2f}\t{3:<10.2f}\t{4:<10.2f}'.format(
        name, res['N Clusters'], res['HC metric'], res['SC metric'],
        res['Time']))
    
print('\nTotal time = {0:.2f}'.format(t_global_final - t_global_start))

#--------------------------------------------------------------------------
#                    Algoritmos Modificados 
#--------------------------------------------------------------------------
birch_modificado = Birch(n_clusters=5,threshold=0.1)
bandwidth = estimate_bandwidth(X_normal, quantile=0.2, random_state=123456)
meanshift_modificado = MeanShift(bandwidth=bandwidth,bin_seeding=True)

clustering_algorithms_modificados = (("Birch_modificado", birch_modificado),
                                     ("meanshift_modificado", meanshift_modificado)
)

t_global_start = time.time()

for algorithm_name,algorithm in clustering_algorithms_modificados:
    results = dict()
    met, clusterFrame, timeAlg,cluster_predict = createPrediction(dataframe=X, data=X_normal, model=algorithm)
    n_clusters=len(set(cluster_predict))

    if( n_clusters > 15 ):
        X_filtrado = clusterFrame[clusterFrame.groupby('cluster').cluster.transform(len) > min_size]
    else:
        X_filtrado = clusterFrame

    makeScatterMatrix(data=X_filtrado,outputName="./imagenes/scatterMatrix_caso3_" +algorithm_name,displayOutput=False)
    makeHeatMapConDendograma(data=X_filtrado,outputName="./imagenes/heatmapcondendograma_caso3_"+algorithm_name,displayOutput=False)
    
    results['N Clusters']=n_clusters
    results['HC metric']=met[0]
    results['SC metric']=met[1]
    results['Time']=timeAlg

    outputData[algorithm_name] = results

t_global_final = time.time()

latexCaso3Modificado = createLatexDataFrame(data=outputData)

f = open('caso3_modificado.txt','w')
f.write(latexCaso3Modificado.to_latex())
f.close()


























