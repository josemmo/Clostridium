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

    def fit(self, z, y, z_tst = None, y_tst = None,  hyper = None, prune = 0, maxit = 15, 
            pruning_crit = 1e-6, tol = 1e-6):
        self.z = z  #(NxK)
        self.z_tst = z_tst  #(NxK_tst)
        self.y_tst = y_tst  #(NxD_tst)
        self.t_tst = y_tst
        self.y = y  #(NxD)
        self.t = y
        self.fact_sel = np.arange(self.z.shape[1])
        #self.K_tr = self.center_K(self.z @ self.z.T)
        self.K_tr = self.z @ self.z.T
        
        self.K = self.z.shape[1] #num dimensiones input
        self.D = self.t.shape[1] #num dimensiones output
        self.N = self.z.shape[0] # num datos
        self.N_tst = self.z_tst.shape[0]
        self.index = np.arange(self.K)
        
        # Some precomputed matrices
        #self.ZTZ = self.z.T @ self.z  #(KxK) es enorme, habría que ver si se puede evitar este calculo
        #self.YTZ = self.y.T @ self.z  #(DxK) 
        self.KTK = self.K_tr.T @ self.K_tr
        self.KTY = self.K_tr.T @ self.y
        self.YTK = self.t.T @ self.K_tr
        self.L = []
        self.mse = []
        self.mse_tst = []        
        self.R2 = []
        self.R2_tst = []
        #self.AUC = []
        #self.AUC_tst = []
        self.K_vec = []
        self.labels_pred = []
        self.input_idx = np.ones(self.K, bool)
        if hyper == None:
            self.hyper = HyperParameters(self.K, self.N)
        else:
            self.hyper = hyper
        self.q_dist = Qdistribution(self.N, self.D, self.K, self.hyper)

        self.fit_vb(prune, maxit, pruning_crit, tol)
        
        
    def center_K(self, K):
        """Center a kernel matrix K, i.e., removes the data mean in the feature space
        Args:
            K: kernel matrix
        """
           
        size_1,size_2 = K.shape;
        D1 = K.sum(axis=0)/size_1
        D2 = K.sum(axis=1)/size_2
        E = D2.sum(axis=0)/size_1
        return K + np.tile(E,[size_1,size_2]) - np.tile(D1,[size_1,1]) - np.tile(D2,[size_2,1]).T


    def pruning(self, pruning_crit):
        q = self.q_dist
        
        maximo = np.max(abs(self.z.T @ q.A['mean']))
        
        self.fact_sel = np.arange(self.z.shape[1])[(abs(self.z.T @ q.A['mean']) > maximo*pruning_crit).flatten()].astype(int)
        
        aux = self.input_idx[self.input_idx]
        aux[self.fact_sel] = False
        self.input_idx[self.input_idx] = ~aux
        
        
        self.z = self.z[:,self.fact_sel]
        self.z_tst = self.z_tst[:,self.fact_sel]
        self.K_tr = self.z @ self.z.T
        q.alpha['a'] = q.alpha['a'][self.fact_sel]
        q.alpha['b'] = q.alpha['b'][self.fact_sel]
        self.hyper.alpha_a = self.hyper.alpha_a[self.fact_sel]
        self.hyper.alpha_b = self.hyper.alpha_b[self.fact_sel]
        self.index = self.index[self.fact_sel]
        

#    def pruning(self, pruning_crit):       
#        q = self.q_dist
#        
#        fact_sel = np.arange(self.z.shape[1])[(abs(np.diagflat(q.V['mean']) @ self.z.T @ q.A['mean'].T) > pruning_crit).flatten()].astype(int)
#        
#        aux = self.input_idx[self.input_idx]
#        aux[fact_sel] = False
#        self.input_idx[self.input_idx] = ~aux
#        
#        # Pruning alpha
#        self.z = self.z[:,fact_sel]
#        self.z_tst = self.z_tst[:,fact_sel]
#        q.V['mean'] = q.V['mean'][fact_sel]
#        q.V['cov'] = q.V['cov'][fact_sel, fact_sel]
#        q.alpha['a'] = q.alpha['a'][fact_sel]
#        q.alpha['b'] = q.alpha['b'][fact_sel]
#        self.hyper.alpha_a = self.hyper.alpha_a[fact_sel]
#        self.hyper.alpha_b = self.hyper.alpha_b[fact_sel]
#        self.index = self.index[fact_sel]
##        q.alpha['a'] = q.alpha['a']
##        q.alpha['b'] = q.alpha['b']
##        self.hyper.alpha_a = self.hyper.alpha_a
##        self.hyper.alpha_b = self.hyper.alpha_b
#        q.K = len(fact_sel)
    
    
    def compute_mse(self, z = None, y = None):
        q = self.q_dist
        if z is None:
            z = self.z_tst
        if y is None:
            y = self.y_tst
        diff = (y - z @ self.z.T @ q.A['mean']).ravel()
        return  diff@diff/self.N
    
    def compute_R2(self, z = None, y = None):
        q = self.q_dist
        if z is None:
            z = self.z_tst
        if y is None:
            y = self.y_tst
        return  r2_score(y.ravel(), (z @ self.z.T @ q.A['mean']).ravel())
        
    def predict(self, Z_test):
        q = self.q_dist
        if Z_test == None:
            Z_test = self.z_tst
        return Z_test @ self.z.T @ q.A['mean']
    
    def predict_proba(self, Z_test):
        q = self.q_dist
        probs = np.zeros((np.shape(Z_test)[0],1))
        for i in range(np.shape(Z_test)[0]):
            mean = Z_test[np.newaxis,i,:] @ self.z.T @ q.A['mean']
            sig = q.tau_mean() + Z_test[np.newaxis,i,:] @ self.z.T @ q.A['cov'] @ self.z @ Z_test[np.newaxis,i,:].T
            prob = self.sigmoid(mean/(np.sqrt(1+(np.pi/8)*sig)))
            probs[i,0] = prob
        return probs
    
    def predict_proba_th(self, Z_test, pruning_crit):
        q = self.q_dist
        maximo = np.max(abs(self.z.T @ q.A['mean']))
        fact = np.arange(self.z.shape[1])[(abs(self.z.T @ q.A['mean']) > maximo*pruning_crit).flatten()].astype(int)
        z = self.z.copy()
        z = z[:,fact]
        Z_test = Z_test[:,fact]

        probs = np.zeros((np.shape(Z_test)[0],1))
        for i in range(np.shape(Z_test)[0]):
            mean = Z_test[np.newaxis,i,:] @ z.T @ q.A['mean']
            sig = q.tau_mean() + Z_test[np.newaxis,i,:] @ z.T @ q.A['cov'] @ z @ Z_test[np.newaxis,i,:].T
            prob = self.sigmoid(mean/(np.sqrt(1+(np.pi/8)*sig)))
            probs[i,0] = prob
        return probs

    
    def return_a(self):
        q = self.q_dist
        return q.A['mean']
    
    def return_alpha(self):
        q = self.q_dist
        return q.alpha_mean()
    
    def return_w(self):
        q = self.q_dist
        return self.z.T @ q.A['mean']
       
    def return_pred(self):
        q = self.q_dist
        
        probs = np.zeros((np.shape(self.z_tst)[0],1))
        sigs = np.zeros((np.shape(self.z_tst)[0],1))
        for i in range(np.shape(self.z_tst)[0]):
            mean = self.z_tst[np.newaxis,i,:] @ self.z.T @ q.A['mean']
            sig = q.tau_mean() + self.z_tst[np.newaxis,i,:] @ self.z.T @ q.A['cov'] @ self.z @ self.z_tst[np.newaxis,i,:].T
            prob = self.sigmoid(mean/(np.sqrt(1+(np.pi/8)*sig)))
            probs[i,0] = prob
            sigs[i,0] = sig
        pred = np.where(probs.ravel()>0.5,1,0)
        return pred, sigs
    
    
    def return_fact_sel(self):
        q = self.q_dist
        return self.fact_sel
    
    def return_index(self):
        q = self.q_dist
        return self.index
    
    def return_proba(self):
        q = self.q_dist
        
        probs = np.zeros((np.shape(self.z_tst)[0],1))
        # inv = self.myInverse(q.A['cov']) 
        for i in range(np.shape(self.z_tst)[0]):
            mean = self.z_tst[np.newaxis,i,:] @ self.z.T @ q.A['mean']
            sig = q.tau_mean() + self.z_tst[np.newaxis,i,:] @ self.z.T @ q.A['cov'] @ self.z @ self.z_tst[np.newaxis,i,:].T
            # sig = 1/q.tau_mean() + self.z_tst[np.newaxis,i,:] @ self.z.T @ q.A['cov'] @ self.z @ self.z_tst[np.newaxis,i,:].T
            prob = self.sigmoid(mean/(np.sqrt(1+(np.pi/8)*sig)))
            probs[i,0] = prob
        return probs
    
    def return_proba_th(self, pruning_crit):
        q = self.q_dist
        maximo = np.max(abs(self.z.T @ q.A['mean']))
        fact = np.arange(self.z.shape[1])[(abs(self.z.T @ q.A['mean']) > maximo*pruning_crit).flatten()].astype(int)
        self.z = self.z[:,fact]
        self.z_tst = self.z_tst[:,fact]

        probs = np.zeros((np.shape(self.z_tst)[0],1))
        for i in range(np.shape(self.z_tst)[0]):
            mean = self.z_tst[np.newaxis,i,:] @ self.z.T @ q.A['mean']
            sig = q.tau_mean() + self.z_tst[np.newaxis,i,:] @ self.z.T @ q.A['cov'] @ self.z @ self.z_tst[np.newaxis,i,:].T
            # sig = 1/q.tau_mean() + self.z_tst[np.newaxis,i,:] @ self.z.T @ q.A['cov'] @ self.z @ self.z_tst[np.newaxis,i,:].T
            prob = self.sigmoid(mean/(np.sqrt(1+(np.pi/8)*sig)))
            probs[i,0] = prob
        return probs


    
    def return_proba_train(self):
        q = self.q_dist
        
        probs = np.zeros((np.shape(self.z)[0],1))
        for i in range(np.shape(self.z)[0]):
            mean = self.z[np.newaxis,i,:] @ self.z.T @ q.A['mean']
            sig = q.tau_mean() + self.z[np.newaxis,i,:] @ self.z.T @ q.A['cov'] @ self.z @ self.z[np.newaxis,i,:].T
            prob = self.sigmoid(mean/(np.sqrt(1+(np.pi/8)*sig)))
            probs[i,0] = prob
        return probs
        
    def sigmoid(self,x):
        if any(x < 0):
            return np.exp(x)/(1 + np.exp(x))
        else:
            return 1/(1 + np.exp(-x))
    
    def fit_vb(self, prune, maxit=30, pruning_crit = 1e-1, tol = 1e-6):
        q = self.q_dist
        for i in range(maxit):
            self.update()

            ##############
            #self.labels_pred.append(self.predict(self.z_tst))
            
            #print('MSE: ',self.compute_mse(self.z_tst,self.y_tst))
            ##############
            print(i)
            self.L.append(self.update_bound())
            print('Features: ',np.shape(self.z)[1])
            if prune == 1 and i>3:
                self.pruning(pruning_crit)
            ##################
            ##################
            #print('\rIteration %d Lower Bound %.1f K %4d' %(i+1, self.L[-1]), end='\r', flush=True)
            if (len(self.L) > 100) and (abs(1 - np.mean(self.L[-101:-1])/self.L[-1]) < tol):
               print('\nModel correctly trained. Convergence achieved')
               return 
            print('LB: ',self.L[-1])               
        print('')

    def update(self):

        self.update_a()
        self.update_alpha()
        self.update_y()
        self.update_xi()
        self.update_tau()

    def myInverse(self,X):
        """Computation of the inverse of a matrix.
        
        This function calculates the inverse of a matrix in an efficient way 
        using the Cholesky decomposition.
        
        Parameters
        ----------
        __A: bool, (default 0). 
            Whether or not to print all the lower bound updates.
            
        """
        
        # try:
        #     L = linalg.pinv(np.linalg.cholesky(X), rcond=1e-10) #np.linalg.cholesky(A)
        #     return np.dot(L.T,L) #linalg.pinv(L)*linalg.pinv(L.T)
        # except:
        #     return np.nan
        try:
            return linalg.pinv(X)
        except:
            return np.nan
        
    
    def update_a(self):
        q = self.q_dist
        
        a_cov = self.z @ np.diagflat(q.alpha_mean()) @ self.z.T + q.tau_mean() * (self.K_tr.T @ self.K_tr)
        a_cov_inv = self.myInverse(a_cov)
        
        if not np.any(np.isnan(a_cov_inv)):
            q.A['cov'] = a_cov_inv
            ##############
            q.A['mean'] = q.tau_mean() * q.A['cov'] @ self.K_tr.T @ q.Y['mean']
            #############
            q.A['prodT'] = q.A['mean'] @ q.A['mean'].T + q.A['cov']
        else:
            print('Covariance of A not invertible, not Updated')
    
    def update_alpha(self):
        q = self.q_dist

        q.alpha['a'] = (self.hyper.alpha_a + 0.5)
        q.alpha['b'] = (self.hyper.alpha_b + 0.5 * np.diag(self.z.T @ q.A['prodT'] @ self.z))
    
    def update_tau(self):
        q = self.q_dist
        
        q.tau['a'] = self.N*0.5 + self.hyper.tau_a
        
        q.tau['b'] = 0.5*(np.trace(q.Y['prodT']) - 2*np.trace(q.Y['mean'].T @ self.K_tr @ q.A['mean']) + np.trace(self.K_tr.T @ self.K_tr @ q.A['prodT']))+self.hyper.tau_b
    
    def update_y(self):
        q = self.q_dist
        y_cov = q.tau_mean()*np.eye(self.N) + 2*np.diagflat(q.xi['mean'])
        y_cov_inv = self.myInverse(y_cov)
        
        if not np.any(np.isnan(y_cov_inv)):
            q.Y['cov'] = y_cov_inv
            
            
            #q.Y['mean'] = q.Y['cov'] @ (self.t - 0.5*np.ones((self.N,1)) + q.tau_mean() * self.z @ self.z.T @ q.A['mean'])

            q.Y['mean'] = q.Y['cov'] @ (self.t - 0.5*np.ones((self.N,1)) + q.tau_mean() * self.K_tr @ q.A['mean'])
            
            q.Y['prodT'] = q.Y['mean'] @ q.Y['mean'].T + q.Y['cov']
        else:
            print('Covariance of Y not invertible, not Updated')
    
    def update_xi(self):
        q = self.q_dist
        q.xi['mean'] = (q.Y['mean'].reshape((self.N,)))**2 + np.diag(q.Y['cov'])
            
    def HGamma(self, a, b):
        """Compute the entropy of a Gamma distribution.

        Parameters
        ----------
        __a: float. 
            The parameter a of a Gamma distribution.
        __b: float. 
            The parameter b of a Gamma distribution.

        """
        
        return -np.log(b)
    
    def HGauss(self, mn, cov, entr):
        """Compute the entropy of a Gamma distribution.
        
        Uses slogdet function to avoid numeric problems. If there is any 
        infinity, doesn't update the entropy.
        
        Parameters
        ----------
        __mean: float. 
            The parameter mean of a Gamma distribution.
        __covariance: float. 
            The parameter covariance of a Gamma distribution.
        __entropy: float.
            The entropy of the previous update. 

        """
        
        H = 0.5*mn.shape[0]*np.linalg.slogdet(cov)[1]
        return self.checkInfinity(H, entr)
        
    def checkInfinity(self, H, entr):
        """Checks if three is any infinity in th entropy.
        
        Goes through the input matrix H and checks if there is any infinity.
        If there is it is not updated, if there isn't it is.
        
        Parameters
        ----------
        __entropy: float.
            The entropy of the previous update. 

        """
        
        if abs(H) == np.inf:
            return entr
        else:
            return H
        
    def update_bound(self):
        """Update the Lower Bound.
        
        Uses the learnt variables of the model to update the lower bound.
        
        """
        
        q = self.q_dist
        q.A['LH'] = self.HGauss(q.A['mean'], q.A['cov'], q.A['LH'])
#        q.V['LH'] = self.HGauss(q.V['mean'], np.diagflat(q.V['cov']), q.V['LH'])
        #lel = self.HGauss(q.V['mean'], np.diagflat(q.V['cov']), q.V['LH'])
        #q.b['LH'] = self.HGauss(q.b['mean'], q.b['cov'], q.b['LH'])
        #q.W['LH'] = self.HGauss(q.W['mean'], q.W['cov'], q.W['LH'])
        #self.W['LH'] = self.z.T @ q.A['LH']
        # Entropy of alpha and tau
        # q.alpha['LH'] = np.sum(self.HGamma(q.alpha['a'], q.alpha['b']))
        # q.tau['LH'] = np.sum(self.HGamma(q.tau['a'], q.tau['b']))
            
        # Total entropy
        # EntropyQ = q.W['LH'] + q.alpha['LH']  + q.tau['LH']
           
        # Calculation of the E[log(p(Theta))]
        
        q.tau['ElogpWalp'] = -(0.5 *  self.N + self.hyper.tau_a - 2)* np.log(q.tau['b'])
        q.alpha['Elogp'] = -(0.5 + np.mean(self.hyper.alpha_a) - 2)* np.sum(np.log(q.alpha['b']))
        
        # Total E[log(p(Theta))]
        ElogP = q.tau['ElogpWalp'] + q.alpha['Elogp'] 
        return ElogP - q.A['LH']
    
#    def update_truncated(self, mu, sig):
#        alfa_fi = (1/(np.sqrt(2*np.pi)))*np.exp((mu**2)/(2*(sig)))
#        alfa_FI = (1/2)*(1 + math.erf(-((mu)/np.sqrt(sig*2))))
#        
#        sigma = sig*(1 + (-((mu/np.sqrt(sig))*alfa_fi)/(1-alfa_FI)) - (alfa_fi/(1-alfa_FI))**2)
#        mean = mu + (alfa_fi/(1-alfa_FI))*sig
#        
#        return mean, sigma
               
    def update_abs(self, mea, sig):
        mean = np.sqrt((sig*2)/(np.pi))*np.exp(-(mea**2)/(2*(sig))) + mea*(1 - 2*sc.stats.norm.cdf(-(mea/np.sqrt(sig))))
        sigma = sig + mea**2 - mean**2
        return mean, sigma
        

class HyperParameters(object):
    def __init__(self, K, N):
        #self.alpha_a = 2 * np.ones((K,))
        #self.alpha_b = 1 * np.ones((K,))
        self.alpha_a = 1000 * np.ones((K,))
        #self.alpha_a = 100000 * np.ones((K,))
        self.alpha_b = 1 * np.ones((K,))

        #self.alpha_a = 0.1 * np.ones((K,))
        #self.alpha_b = 0.01 * np.ones((K,))

        #self.alpha_a = 1e-12 * np.ones((K,))
        #self.alpha_b = 1e-14 * np.ones((K,))

        #self.tau_a = 1e-12
        #self.tau_b = 1e-14
        self.tau_a = 1e-14
        self.tau_b = 1e-14


class Qdistribution(object):
    def __init__(self, n, D, K, hyper):
        self.n = n
        self.D = D
        self.K = K
        
        # Initialize gamma disributions
        alpha = self.qGamma(hyper.alpha_a,hyper.alpha_b,self.K)
        self.alpha = alpha 
        tau = self.qGamma(hyper.tau_a,hyper.tau_b,1)
        self.tau = tau 

        # The remaning parameters at random
        self.init_rnd()

    def init_rnd(self):
        self.A = {
                "mean":     None,
                "cov":      None,
                "prodT":    None,
                "LH":       0,
                "Elogp":    0,
            }
        
        self.Y = {
                "mean":     None,
                "cov":      None,
                "prodT":    None,
                "LH":       0,
                "Elogp":    0,
            }
        
        self.xi = {
                "mean":     None,
                "cov":      None,
                "prodT":    None,
                "LH":       0,
                "Elogp":    0,
            }
            
#        self.W["mean"] = np.random.normal(0.0, 1.0, self.D * self.K).reshape(self.D, self.K)
#        self.W["cov"] = np.eye(self.K)
#        self.W["prodT"] = np.dot(self.W["mean"].T, self.W["mean"])+self.K*self.W["cov"]
        
        
        
        #self.A["mean"] = np.random.normal(0.0, 1.0, self.D * self.K).reshape(self.D, self.K)
        #np.random.seed(41)
        self.A["mean"] = np.random.normal(0.0, 1.0, self.n).reshape(self.n,1)
        #self.A["cov"] = np.eye(self.K)
        self.A["cov"] = np.eye(self.n)
        #self.A["prodT"] = np.dot(self.A["mean"].T, self.A["mean"])+self.K*self.A["cov"]
        self.A["prodT"] = self.A["mean"].T @ self.A["mean"]+self.n*self.A["cov"]
        
        #Inicializamos las Y
        self.Y["mean"] = (1/2)*np.ones((self.n,1))
        self.Y["cov"] = np.eye(self.n)
        self.Y["prodT"] = self.Y["mean"].T @ self.Y["mean"]+self.n*self.Y["cov"]
        
        #Inicializamos Xi
        self.xi["mean"] = np.ones((self.n,1))
        

    def qGamma(self,a,b,K):
        """ Initialisation of variables with Gamma distribution..
    
        Parameters
        ----------
        __a : array (shape = [1, 1]).
            Initialistaion of the parameter a.        
        __b : array (shape = [K, 1]).
            Initialistaion of the parameter b.
        __m_i: int.
            Number of views. 
        __K: array (shape = [K, 1]).
            dimension of the parameter b for each view.
            
        """
        
        param = {                
                "a":         a,
                "b":         b,
                "LH":         None,
                "ElogpWalp":  None,
            }
        return param
        
    
    def alpha_mean(self):
        return self.alpha['a'] / self.alpha['b'] 
    def tau_mean(self):
        return self.tau['a'] / self.tau['b']

    
