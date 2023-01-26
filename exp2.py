import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy
from sklearn.linear_model import LogisticRegression, LinearRegression
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import train_test_split
from general import *
     
def get_pval_rbpt(X, Z, Y, H, g1, loss='mse'):
    n = X.shape[0]
    XZ = np.hstack((X, Z))
    loss1 = get_loss(Y, g1.predict(X,Z).reshape((-1,1)), loss=loss)
    loss2 = get_loss(Y, H.reshape((-1,1)), loss=loss)
    T = loss2-loss1
    pval = 1 - scipy.stats.norm.cdf(np.sqrt(n)*np.mean(T)/np.std(T))
    return pval

def get_pval_rbpt2(X, Z, Y, g1, h, loss='mae'):
    n = X.shape[0]
    XZ = np.hstack((X, Z))
    loss1 = get_loss(Y, g1.predict(X,Z).reshape((-1,1)), loss=loss)
    loss2 = get_loss(Y, h.predict(Z).reshape((-1,1)), loss=loss)
    T = loss2-loss1
    pval = 1 - scipy.stats.norm.cdf(np.sqrt(n)*np.mean(T)/np.std(T))
    return pval
        
def get_h(X, y, validation_split=.1, verbose=False, random_state=None):
  
    ### Paramaters
    early_stopping_rounds=10
    loss='MultiRMSE'
    
    ### Validating
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split, random_state=random_state)

    m = CatBoostRegressor(loss_function = loss,
                          eval_metric = loss,
                          thread_count=-1,
                          random_seed=random_state)

    m.fit(X_train, y_train, verbose=verbose,
          eval_set=(X_val, y_val),
          early_stopping_rounds = early_stopping_rounds)
    
    
    ### Final model
    m2 = CatBoostRegressor(iterations=int(m.tree_count_), 
                           loss_function = loss,
                           eval_metric = loss,
                           thread_count=-1,
                           random_seed=random_state)

    m2.fit(X, y, verbose=verbose) 
    
    return m2