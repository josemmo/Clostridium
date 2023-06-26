# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 17:06:31 2023

@author: Albert
"""

import numpy as np
from scipy import linalg
from sklearn.metrics import r2_score, auc, roc_curve
import multiprocessing as mp
import math
import scipy as sc

class LR_ARD(object):
    def __init__(self):
        pass

    def fit(self, X, A_mean1, A_mean2,  A_mean3, A_cov1, A_cov2, A_cov3, tau1, tau2, tau3, prune, maximo1, maximo2, maximo3):
        self.prune = prune
        self.maximo1 = maximo1
        self.maximo2 = maximo2
        self.maximo3 = maximo3
        self.X = X
        self.A_mean1 = A_mean1
        self.A_mean2 = A_mean2
        self.A_mean3 = A_mean3
        self.A_cov1 = A_cov1
        self.A_cov2 = A_cov2
        self.A_cov3 = A_cov3
        self.tau1 = tau1
        self.tau2 = tau2
        self.tau3 = tau3
        
        
        
    def predict(self, Z_tst):
        ones = np.ones((np.shape(Z_tst)[0],1))
        Z_tst = np.hstack((Z_tst,ones))

        probs = self.predict_proba_true(Z_tst)
        preds = np.argmax(probs, axis = 1)
        return preds


    def predict_proba_true(self, Z_tst):
        ones = np.ones((np.shape(Z_tst)[0],1))
        Z_tst = np.hstack((Z_tst,ones))

        probs1 = self.predict_proba_one_class(Z_tst,self.X, self.A_mean1, self.A_cov1, self.tau1, self.prune, self.maximo1)
        probs2 = self.predict_proba_one_class(Z_tst,self.X, self.A_mean2, self.A_cov2, self.tau2, self.prune, self.maximo2)
        probs3 = self.predict_proba_one_class(Z_tst,self.X, self.A_mean3, self.A_cov3, self.tau3, self.prune, self.maximo3)

        prob_p = np.hstack((probs1,probs2))
        probs = np.hstack((prob_p,probs3))
        return probs


    def predict_proba(self, Z_tst):
        #Ojo que esto es cutre cutre

        #Calculamos el minimo y maximo de la prob con las salidas de los datos de train
        probs1 = self.predict_proba_one_class(self.X,self.X, self.A_mean1, self.A_cov1, self.tau1, self.prune, self.maximo1)
        probs2 = self.predict_proba_one_class(self.X,self.X, self.A_mean2, self.A_cov2, self.tau2, self.prune, self.maximo2)
        probs3 = self.predict_proba_one_class(self.X,self.X, self.A_mean3, self.A_cov3, self.tau3, self.prune, self.maximo3)

        prob_p = np.hstack((probs1,probs2))
        probs = np.hstack((prob_p,probs3))
        #print(probs)

        maximo = np.max(probs.ravel())
        minimo = np.min(probs.ravel())
        #print('Maximo: ', maximo)
        #print('Minimo: ', minimo)
        #Calculamos las probs del test
        ones = np.ones((np.shape(Z_tst)[0],1))
        Z_tst = np.hstack((Z_tst,ones))

        probs1 = self.predict_proba_one_class(Z_tst,self.X, self.A_mean1, self.A_cov1, self.tau1, self.prune, self.maximo1)
        probs2 = self.predict_proba_one_class(Z_tst,self.X, self.A_mean2, self.A_cov2, self.tau2, self.prune, self.maximo2)
        probs3 = self.predict_proba_one_class(Z_tst,self.X, self.A_mean3, self.A_cov3, self.tau3, self.prune, self.maximo3)

        prob_p = np.hstack((probs1,probs2))
        probs = np.hstack((prob_p,probs3))
        #Normalizamos las probabilidades de salida respecto a los datos de train

        probs_norm = self.normalize_data(probs, maximo, minimo)

        #Chequeamos que ningun casi se ha salido de [0,1]
        probs_norm = np.where(probs_norm > 1.0, 1.0, probs_norm)
        probs_norm = np.where(probs_norm < 0.0, 0.0, probs_norm)
        return probs_norm

    

    def predict_proba_one_class(self, Z_test, X, A_mean, A_cov, tau, prune, maximo):
        fact = np.arange(X.shape[1])[(abs(X.T @ A_mean) > maximo*prune).flatten()].astype(int)
        X = X[:,fact]
        Z_test = Z_test[:,fact]
        mean = Z_test @ X.T @ A_mean
        sig = np.diag(tau + Z_test @ X.T @ A_cov @ X @ Z_test.T).reshape(-1,1)
        probs = self.sigmoid(mean/(np.sqrt(1+(np.pi/8)*sig)))

        return probs
    
    def normalize_data(self,X, maximo, minimo):
        return (X - minimo)/(maximo - minimo)
    
    def sigmoid(self,x):
        if any(x < 0):
            return np.exp(x)/(1 + np.exp(x))
        else:
            return 1/(1 + np.exp(-x))


    
