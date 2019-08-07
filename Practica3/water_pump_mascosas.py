# -*- coding: utf-8 -*-
"""
Autor:
    Felix Ramirez Garcia
Fecha:
    Noviembre/2018
Contenido:
    Practica 3 , competicion en DrivenData:
       https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/
    Inteligencia de Negocio
    Grado en Ingeniería Informática 4º curso
    Universidad de Granada
"""
'''
##############################################
                IMPORTS
##############################################
'''

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score
import time
import lightgbm as lgb
#Silenciar los Warings
import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

'''
##############################################
                FUNCIONES
##############################################
'''
def validacion_cruzada(modelo, X, y, cv):
    y_test_all = []
    
    for train, test in cv.split(X, y):
        t = time.time()
        modelo = modelo.fit(X[train],y[train])
        tiempo = time.time() - t
        y_pred = modelo.predict(X[test])
        print("Score: {:.4f}, tiempo: {:6.2f} segundos".format(accuracy_score(y[test],y_pred) , tiempo))
        y_test_all = np.concatenate([y_test_all,y[test]])

    print("")

    return modelo, y_test_all

'''
##############################################
            LECTURA DE DATOS
##############################################
'''
df_tra=pd.read_csv('training_set_values.csv',parse_dates=True)
data_y=pd.read_csv('training_set_labels.csv')
df_tst=pd.read_csv('test_set_values.csv',parse_dates=True)

carpeta_datos="C:/Users/felix/Desktop/Google-Drive/IN/Practica3/dataset/"
df_submission = pd.read_csv(carpeta_datos+'water_pump_submissionformat.csv')
'''
##############################################
            PREPROCESAMIENTO
##############################################
'''

df=df_tra.append(df_tst)

df.construction_year=pd.to_numeric(df_tra.construction_year)
df.loc[df.construction_year <= 0, df.columns=='construction_year'] = 1950


df['operation_time']=df.date_recorded.apply(pd.to_datetime)-df.construction_year.apply(lambda x: pd.to_datetime(x,format='%Y'))
df['operation_time']=df.operation_time.apply(lambda x: x.total_seconds())

df['month']=pd.to_datetime(df.date_recorded).dt.month

mode_funder = df['funder'].mode()[0]
df['funder'] = df['funder'].fillna(mode_funder)
mode_installer = df['installer'].mode()[0]
df['installer'] = df['installer'].fillna(mode_installer)
mode_subvillage = df['subvillage'].mode()[0]
df['subvillage'] = df['subvillage'].fillna(mode_subvillage)
mode_scheme_management = df['scheme_management'].mode()[0]
df['scheme_management'] = df['scheme_management'].fillna(mode_scheme_management)
mode_scheme_name = df['scheme_name'].mode()[0]
df['scheme_name'] = df['scheme_name'].fillna(mode_scheme_name)

factorschange=[x for x in df.columns if x not in [
        'id',
        'latitude',
        'longitude',
        'gps_height',
        'date_recorded',
        'construction_year',
        'month',
        'operation_time',
        'installer']]

for factor in factorschange:
    values_factor=df[factor].value_counts()
    lessthen=values_factor[values_factor < 20]
    listnow=df.installer.isin(list(lessthen.keys()))
    df.loc[listnow,factor] = 'Others'
    df[factor] = preprocessing.LabelEncoder().fit_transform(df[factor])


df.population = df.population.apply(lambda x: np.log10(x+1))
   
a = df[df["longitude"] < 1]
a.iloc[:,df.columns == "latitude"]= np.nan
a.iloc[:,df.columns == "longitude"]= np.nan
df[df["longitude"] < 1] = a
df["longitude"] = df.groupby("region_code").transform(lambda x: x.fillna(x.mean())).longitude
df["latitude"] = df.groupby("region_code").transform(lambda x: x.fillna(x.mean())).latitude

a = df[df["gps_height"] < 1]
a.iloc[:,df.columns == "gps_height"]= np.nan
df[df["gps_height"] < 1] = a
df["gps_height"] = df.groupby("region_code").transform(lambda x: x.fillna(x.mean())).gps_height

df=df.fillna(df.mean())

dataframe_tst = df[len(df_tra):]
dataframe_tra = df[:len(df_tra)]

print("--------TST-----------")
#print(dataframe_tst)
print("--------TRA-----------")
#print(dataframe_tra)

dataframe_tst.drop(labels=['id'], axis=1,inplace = True)
dataframe_tst.drop(labels=['date_recorded'], axis=1,inplace = True)
dataframe_tst.drop(labels=['longitude'], axis=1,inplace = True)
dataframe_tst.drop(labels=['latitude'], axis=1,inplace = True)
dataframe_tst.drop(labels=['wpt_name'], axis=1,inplace = True)
dataframe_tst.drop(labels=['num_private'], axis=1,inplace = True)

dataframe_tra.drop(labels=['id'], axis=1,inplace = True)
dataframe_tra.drop(labels=['date_recorded'], axis=1,inplace = True)
dataframe_tra.drop(labels=['longitude'], axis=1,inplace = True)
dataframe_tra.drop(labels=['latitude'], axis=1,inplace = True)
dataframe_tra.drop(labels=['wpt_name'], axis=1,inplace = True)
dataframe_tra.drop(labels=['num_private'], axis=1,inplace = True)

data_y.drop(labels=['id'], axis=1,inplace = True)

######################### TRAINING ##############################
mask = dataframe_tra.isnull()  #máscara para luego recuperar los NaN
data_x_tmp = dataframe_tra.fillna(9999) # Combierte los NaN en 9999
data_x_tmp = data_x_tmp.astype(str).apply(LabelEncoder().fit_transform) #se convierten categóricas en numéricas
data_x_tra_nan = data_x_tmp.where(~mask, dataframe_tra)  #se recuperan los NaN

############################## TEST ##############################
mask = dataframe_tst.isnull() #máscara para luego recuperar los NaN
data_x_tmp = dataframe_tst.fillna(9999) #LabelEncoder no funciona con NaN, se asigna un valor no usado
data_x_tmp = data_x_tmp.astype(str).apply(LabelEncoder().fit_transform) #se convierten categóricas en numéricas
data_x_tst_nan = data_x_tmp.where(~mask, dataframe_tst) #se recuperan los NaN

dataframe_tra = data_x_tra_nan.values
dataframe_tst = data_x_tst_nan.values
y = np.ravel(data_y.values)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123456)

modelRFC = RandomForestClassifier(n_estimators=1000,min_samples_split=10,criterion='gini')
modelRFC, y_test_modelRFC = validacion_cruzada(modelRFC,dataframe_tra,y,skf)
modelRFC = modelRFC.fit(dataframe_tra,y)
y_pred_tra_modelRFC = modelRFC.predict(dataframe_tra)
print("Score Random: {:.4f}".format(accuracy_score(y,y_pred_tra_modelRFC)))
y_pred_tst_modelRFC = modelRFC.predict(dataframe_tst)
df_submission['status_group'] = y_pred_tst_modelRFC
df_submission.to_csv("submission_modelRFC.csv", index=False)


#modelETC = ExtraTreesClassifier(n_estimators=1000,min_samples_split=10)
#modelETC, y_test_modelETC = validacion_cruzada(modelETC,dataframe_tra,y,skf)
#modelETC = modelETC.fit(dataframe_tra,y)
#y_pred_tra_modelETC = modelETC.predict(dataframe_tra)
#print("Score Random: {:.4f}".format(accuracy_score(y,y_pred_tra_modelETC)))
#y_pred_tst_modelETC = modelETC.predict(dataframe_tst)
#df_submission['status_group'] = y_pred_tst_modelETC
#df_submission.to_csv("submission_modelETC.csv", index=False)

#print("------ LightGBM ------")
#light = lgb.LGBMClassifier(feature_fraction=0.3,learning_rate=0.01,n_estimators=2500,num_leaves=200)
#ligth, y_test_lgbm = validacion_cruzada(light,dataframe_tra,y,skf)
#ligth = ligth.fit(dataframe_tra,y)
#y_pred_tra_ligth = ligth.predict(dataframe_tra)
#print("Score LightGBM: {:.4f}".format(accuracy_score(y,y_pred_tra_ligth)))
#y_pred_tst_ligth = ligth.predict(dataframe_tst)
#df_submission['status_group'] = y_pred_tst_ligth
#df_submission.to_csv("submission_ligth.csv", index=False)













