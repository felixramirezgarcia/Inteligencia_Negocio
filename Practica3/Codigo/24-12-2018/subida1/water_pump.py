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
###########################################################
                              IMPORTS
###########################################################
'''
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

'''
############################################################
                            FUNCIONES
############################################################
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
############################################################
                        LECTURA DE DATOS
############################################################
'''
#los ficheros .csv se han preparado previamente para sustituir ,, y "Not known" por NaN (valores perdidos)
carpeta_datos="C:/Users/felix/Desktop/Google_Drive/IN/Practica3/dataset/"
# Datos training
data_x_tra = pd.read_csv(carpeta_datos+'water_pump_tra.csv',na_values="unknown")
# La clasificacion
data_y = pd.read_csv(carpeta_datos+'water_pump_tra_target.csv')
# Datos test
data_x_tst = pd.read_csv(carpeta_datos+'water_pump_tst.csv',na_values="unknown")

'''
############################################################
                    ELIMINACION DE COLUMNAS
############################################################
'''
################# eliminar id ###################################
data_x_tra.drop(labels=['id'], axis=1,inplace = True)
data_x_tra.drop(labels=['date_recorded'], axis=1,inplace = True)
data_x_tst.drop(labels=['id'], axis=1,inplace = True)
data_x_tst.drop(labels=['date_recorded'], axis=1,inplace = True)
data_y.drop(labels=['id'], axis=1,inplace = True)
    
'''
############################################################
                     CATEGORICAS A NUMERICAS
############################################################
'''
######################### TRAINING ##############################
mask = data_x_tra.isnull()  #máscara para luego recuperar los NaN
data_x_tmp = data_x_tra.fillna(9999) # Combierte los NaN en 9999
data_x_tmp = data_x_tmp.astype(str).apply(LabelEncoder().fit_transform) #se convierten categóricas en numéricas
data_x_tra_nan = data_x_tmp.where(~mask, data_x_tra)  #se recuperan los NaN

############################# TEST ##############################
mask = data_x_tst.isnull() #máscara para luego recuperar los NaN
data_x_tmp = data_x_tst.fillna(9999) #LabelEncoder no funciona con NaN, se asigna un valor no usado
data_x_tmp = data_x_tmp.astype(str).apply(LabelEncoder().fit_transform) #se convierten categóricas en numéricas
data_x_tst_nan = data_x_tmp.where(~mask, data_x_tst) #se recuperan los NaN

X_tra = data_x_tra_nan.values
X_tst = data_x_tst_nan.values
y = np.ravel(data_y.values)
'''
############################################################
                             ALGORITMOS
############################################################
'''
#Validación cruzada con particionado estratificado y control de la aleatoriedad fijando la semilla
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123456)
df_submission = pd.read_csv(carpeta_datos+'water_pump_submissionformat.csv')

#------------------------------ XGB ---------------------------------------
print("------ XGB ------")
gradient = xgb.XGBClassifier(n_estimators = 200)
gradient,y_test_gradient = validacion_cruzada(gradient,X_tra,y,skf)
gradient = gradient.fit(X_tra,y)
y_pred_tra_gradient = gradient.predict(X_tra)
print("Score XGB: {:.4f}".format(accuracy_score(y,y_pred_tra_gradient)))
y_pred_tst_gradient = gradient.predict(X_tst)
df_submission['status_group'] = y_pred_tst_gradient
df_submission.to_csv("submission_gradient.csv", index=False)

#----------------------------- LightGBM -------------------------------------
print("------ LightGBM ------")
light = lgb.LGBMClassifier(objective='binary',n_estimators=200,num_threads=2)
ligth, y_test_lgbm = validacion_cruzada(light,X_tra,y,skf)
ligth = ligth.fit(X_tra,y)
y_pred_tra_ligth = ligth.predict(X_tra)
print("Score LightGBM: {:.4f}".format(accuracy_score(y,y_pred_tra_ligth)))
y_pred_tst_ligth = ligth.predict(X_tst)
df_submission['status_group'] = y_pred_tst_ligth
df_submission.to_csv("submission_ligth.csv", index=False)





















