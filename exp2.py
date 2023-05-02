import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy
from sklearn.linear_model import LogisticRegression, LinearRegression
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
import copy
import time
from general import *
from exp1 import get_pval_stfr, get_pval_crt

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

###CRT
def sample_crt(Z, model):
    return 1*(model.predict_proba(Z)[:,1]>np.random.uniform(0,1,Z.shape[0])).reshape((-1,1))

def get_pval_crt(X, Z, Y, p_model, g1, g2, B=500, loss='mae'):
    n = X.shape[0]
    loss1 = get_loss(Y, g1.predict(X,Z))
    loss2 = get_loss(Y, g2.predict(None,Z))
    T = np.mean(loss2-loss1)
    
    Ts=[]
    for i in range(B):
        x = sample_crt(Z, p_model)
        loss1 = get_loss(Y, g1.predict(x,Z))
        Ts.append(np.mean(loss2-loss1))
    
    Ts=np.array(Ts)
    pval=(1+np.sum(Ts>=T))/(B+1)
    return pval

### CPT
### Code extracted from https://rss.onlinelibrary.wiley.com/doi/pdf/10.1111/rssb.12340
# generate CPT copies of X when the conditional distribution is Bernoulli
# i.e. X | Z=Z_i ~ Ber(mu[i])
def generate_X_CPT_bernoulli(nstep,M,X0,mu):
    log_lik_mat = X0[:,None]*np.log(mu[None,:]) + (1-X0[:,None])*np.log(1-mu[None,:])
    Pi_mat = generate_X_CPT(nstep,M,log_lik_mat)
    return X0[Pi_mat]

# generate CPT copies of X in general case
# log_lik_mat[i,j] = q(X[i]|Z[j]) where q(x|z) is the conditional density for X|Z
def generate_X_CPT(nstep,M,log_lik_mat,Pi_init=[]):
    n = log_lik_mat.shape[0]
    if len(Pi_init)==0:
        Pi_init = np.arange(n,dtype=int)
    Pi_ = generate_X_CPT_MC(nstep,log_lik_mat,Pi_init)
    Pi_mat = np.zeros((M,n),dtype=int)
    for m in range(M):
        Pi_mat[m] = generate_X_CPT_MC(nstep,log_lik_mat,Pi_)
    return Pi_mat
def generate_X_CPT_MC(nstep,log_lik_mat,Pi):
    n = len(Pi)
    npair = np.floor(n/2).astype(int)
    for istep in range(nstep):
        perm = np.random.choice(n,n,replace=False)
        inds_i = perm[0:npair]
        inds_j = perm[npair:(2*npair)]
        # for each k=1,...,npair, decide whether to swap Pi[inds_i[k]] with Pi[inds_j[k]]
        log_odds = log_lik_mat[Pi[inds_i],inds_j] + log_lik_mat[Pi[inds_j],inds_i] \
            - log_lik_mat[Pi[inds_i],inds_i] - log_lik_mat[Pi[inds_j],inds_j]
        swaps = np.random.binomial(1,1/(1+np.exp(-np.maximum(-500,log_odds))))
        Pi[inds_i], Pi[inds_j] = Pi[inds_i] + swaps*(Pi[inds_j]-Pi[inds_i]), Pi[inds_j] - \
            swaps*(Pi[inds_j]-Pi[inds_i])   
    return Pi

#The next function is our own code
def get_pval_cpt(X, Z, Y, p_model, g1, g2, B=500, loss='mae'):
    
    n = X.shape[0]
    loss1 = get_loss(Y, g1.predict(X,Z))
    loss2 = get_loss(Y, g2.predict(None,Z))
    T = np.mean(loss2-loss1)
    
    nstep=50
    mu=p_model.predict_proba(Z)[:,1]
    sig2=np.ones(X.shape[0])
    X_CPT = generate_X_CPT_bernoulli(nstep=nstep, M=B, X0=X.squeeze(), mu=mu)

    Ts=[]
    for i in range(B):
        x = X_CPT[i,:].reshape((-1,1))
        loss1 = get_loss(Y, g1.predict(x,Z))
        Ts.append(np.mean(loss2-loss1))
    
    Ts=np.array(Ts)
    pval=(1+np.sum(Ts>=T))/(B+1)
    return pval

###Regression
def get_pval_gcm(X, Z, Y, g2, p_model):
    n = X.shape[0]
    rx = X-p_model.predict_proba(Z)[:,1].reshape(X.shape)
    ry = Y-g2.predict(None, Z)
    T = rx.squeeze()*ry.squeeze()
    pval = 2*(1 - scipy.stats.norm.cdf(abs(np.sqrt(n)*np.mean(T)/np.std(T))))
    return pval

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


#####
def exp22(it, n_vals, loss, alpha, B, percent=True):
    
    states=['ca','il','mo','tx']
    pvals=[]
    times=[]
    count=0

    for s in states:

        data = pd.read_csv('data/car-insurance-public/data/' + s + '-per-zip.csv')
        companies = list(set(data.companies_name))

        for cia in companies:

            data = pd.read_csv('data/car-insurance-public/data/' + s + '-per-zip.csv')
            data = data.loc[:,['state_risk','combined_premium','minority','companies_name']].dropna()
            data = data.loc[data.companies_name == cia]

            Z = np.array(data.state_risk).reshape((-1,1))
            Y = np.array(data.combined_premium).reshape((-1,1))
            X = (1*np.array(data.minority)).reshape((-1,1))
    
            if percent:
                bins = np.percentile(Z, np.linspace(0,100,n_vals+2))
            else:
                bins = np.linspace(np.min(Z),np.max(Z),n_vals+2)
            bins = bins[1:-1]
            Y_ci = copy.deepcopy(Y)
            Z_bin = np.array([find_nearest(bins, z) for z in Z.squeeze()]).reshape(Z.shape)

            for val in np.unique(Z_bin):
                ind = Z_bin==val
                rng = np.random.RandomState(it)
                ind2 = rng.choice(np.sum(ind),np.sum(ind),replace=False)
                Y_ci[ind] = Y_ci[ind][ind2]

            X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(X, Y_ci, Z_bin, test_size=.3, random_state=it)

            #print(np.min([np.sum(Z_train.squeeze()==val) for val in np.unique(Z_train)]),
            #      np.min([np.sum(Z_test.squeeze()==val) for val in np.unique(Z_test)]))

            ###Fitting models
            g1 = g()
            g1.fit(X_train, Z_train, Y_train)
            g2 = g()
            g2.fit(None, Z_train, Y_train)
          
            ###RBPT
            start_time = time.time()
            p = LogisticRegressionCV(cv=5, scoring='neg_log_loss', solver='liblinear', random_state=0).fit(Z_train, X_train.squeeze())
            H_test = np.sum(p.predict_proba(Z_test)*np.hstack((g1.predict(np.zeros(X_test.shape),Z_test).reshape(-1,1),
                                                               g1.predict(np.ones(X_test.shape),Z_test).reshape(-1,1))), axis=1).reshape(-1,1)
            pval_rbpt = get_pval_rbpt(X_test, Z_test, Y_test, H_test, g1, loss=loss)
            time_rbpt = time.time() - start_time

            ###RBPT2
            start_time = time.time()
            h = get_h(Z_train, g1.predict(X_train,Z_train).squeeze())
            pval_rbpt2 = get_pval_rbpt2(X_test, Z_test, Y_test, g1, h, loss=loss)
            time_rbpt2 = time.time() - start_time

            ###STFR
            start_time = time.time()
            pval_stfr = get_pval_stfr(X_test, Z_test, Y_test, g1, g2, loss=loss)
            time_stfr = time.time() - start_time

            ###GCM
            start_time = time.time()
            pval_gcm = get_pval_gcm(X_test, Z_test, Y_test, g2, p) 
            time_gcm = time.time() - start_time

            ###CRT
            start_time = time.time()
            pval_crt = get_pval_crt(X_test, Z_test, Y_test, p, g1, g2, B=B, loss=loss)
            time_crt = time.time() - start_time

            ###CPT
            start_time = time.time()
            pval_cpt = get_pval_cpt(X_test, Z_test, Y_test, p, g1, g2, B=B, loss=loss)
            time_cpt = time.time() - start_time

            ###Storing
            times.append([count, time_rbpt, time_rbpt2, time_stfr, time_gcm, time_crt, time_cpt])
            pvals.append([count, pval_rbpt, pval_rbpt2, pval_stfr, pval_gcm, pval_crt, pval_cpt])

        count+=1

    pvals = np.array(pvals)
    pvals[:,1:] = 1*(pvals[:,1:]<=alpha)
    reject_prop = np.array(pd.DataFrame(pvals).groupby(by=[0]).mean()).tolist()

    return reject_prop, times