import numpy as np
import scipy
from scipy import stats
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from matplotlib.ticker import FormatStrFormatter
from sklearn.feature_selection import mutual_info_regression
import matplotlib.colors as colors
import time

from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures


from general import *

left  = 0.125  # the left side of the subplots of the figure
right = 0.9    # the right side of the subplots of the figure
bottom = 0.1   # the bottom of the subplots of the figure
top = 0.9      # the top of the subplots of the figure
wspace = 0.3   # the amount of width reserved for blank space between subplots
hspace = 0.3   # the amount of height reserved for white space between subplots

def get_pval_stfr(X, Z, Y, g1, g2, loss='mae'):
    n = X.shape[0]
    loss1 = get_loss(Y, g1.predict(X,Z), loss=loss)
    loss2 = get_loss(Y, g2.predict(None,Z), loss=loss)
    T = loss2-loss1
    pval = 1 - scipy.stats.norm.cdf(np.sqrt(n)*np.mean(T)/np.std(T))
    return pval

def get_pval_gcm(X, Z, Y, g2, g3):
    n = X.shape[0]
    rx = X-g3.predict(None, Z)
    ry = Y-g2.predict(None, Z)
    T = rx.squeeze()*ry.squeeze()
    pval = 2*(1 - scipy.stats.norm.cdf(abs(np.sqrt(n)*np.mean(T)/np.std(T))))
    return pval

def get_pval_resit(X, Z, Y, g2, g3, B=-1):
    rx = X-g3.predict(None, Z)
    ry = Y-g2.predict(None, Z)

    mi = stats.spearmanr(rx.squeeze(), ry.squeeze())[0] #mutual_info_regression(rx, ry.squeeze())
    mis = []
    
    if B==-1:
        pval=stats.spearmanr(rx.squeeze(), ry.squeeze())[1]
        
    else:
        for b in range(B):
            np.random.shuffle(rx)
            mis.append(stats.spearmanr(rx.squeeze(), ry.squeeze())[0])
        mis = np.array(mis)
        pval = np.mean((1+np.sum(mi<=mis))/(B+1))
    
    return pval

### CRT
def sample_crt(Z, b, theta):
    mu=mu_x(Z, b) + theta
    return np.random.normal(mu,1).reshape(-1,1)

def get_pval_crt(X, Z, Y, b, g1, g2, theta, B=500, loss='mae'):
    n = X.shape[0]
    loss1 = get_loss(Y, g1.predict(X,Z))
    loss2 = get_loss(Y, g2.predict(None,Z))
    T = np.mean(loss2-loss1)
    
    Ts=[]
    for i in range(B):
        x = sample_crt(Z, b, theta)
        loss1 = get_loss(Y, g1.predict(x,Z))
        Ts.append(np.mean(loss2-loss1))
    
    Ts=np.array(Ts)
    pval=(1+np.sum(Ts>=T))/(B+1)
    return pval

#### RBPT
#def predict_h(Z, cd, x, g1):
#    y = np.hstack([g1.predict(x[:,j:(j+1)],Z) for j in range(x.shape[1])])
#    y = y * cd
#    return np.trapz(y, x, axis=1).reshape(-1,1)

#def get_pval_rbpt(X, Z, Y, g1, cd, x_grid, loss='mae'):
#    n = X.shape[0]
#    loss1 = get_loss(Y, g1.predict(X,Z), loss=loss)
#    loss2 = get_loss(Y, predict_h(Z, cd, x_grid, g1), loss=loss)
#    T = loss2-loss1
#    pval = 1 - scipy.stats.norm.cdf(np.sqrt(n)*np.mean(T)/np.std(T))
#    return pval

def get_pval_rbpt(X, Z, Y, b, g1, theta, loss='mae'):
    n = X.shape[0]
    loss1 = get_loss(Y, g1.predict(X,Z), loss=loss)
    loss2 = get_loss(Y, g1.predict(mu_x(Z, b) + theta,Z), loss=loss)
    T = loss2-loss1
    pval = 1 - scipy.stats.norm.cdf(np.sqrt(n)*np.mean(T)/np.std(T))
    return pval

def get_pval_rbpt2(X, Z, Y, g1, h, loss='mae'):
    n = X.shape[0]
    loss1 = get_loss(Y, g1.predict(X,Z), loss=loss)
    loss2 = get_loss(Y, h.predict(Z).reshape((-1,1)), loss=loss)
    T = loss2-loss1
    pval = 1 - scipy.stats.norm.cdf(np.sqrt(n)*np.mean(T)/np.std(T))
    return pval



### CPT
### Code extracted from https://rss.onlinelibrary.wiley.com/doi/pdf/10.1111/rssb.12340
# generate CPT copies of X when the conditional distribution is Gaussian
# i.e. X | Z=Z_i ~ N(mu[i],sig2[i])
def generate_X_CPT_gaussian(nstep,M,X0,mu,sig2):
    log_lik_mat = - np.power(X0,2)[:,None] * (1/2/sig2)[None,:] + X0[:,None] * (mu/sig2)[None,:]
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
def get_pval_cpt(X, Z, Y, b, g1, g2, theta, B=500, loss='mae'):
    
    n = X.shape[0]
    loss1 = get_loss(Y, g1.predict(X,Z))
    loss2 = get_loss(Y, g2.predict(None,Z))
    T = np.mean(loss2-loss1)
    
    nstep=50
    mu=(mu_x(Z, b) + theta).squeeze()
    sig2=np.ones(X.shape[0])
    X_CPT = generate_X_CPT_gaussian(nstep=nstep, M=B, X0=X.squeeze(), mu=mu, sig2=sig2)

    Ts=[]
    for i in range(B):
        x = X_CPT[i,:].reshape((-1,1))
        loss1 = get_loss(Y, g1.predict(x,Z))
        Ts.append(np.mean(loss2-loss1))
    
    Ts=np.array(Ts)
    pval=(1+np.sum(Ts>=T))/(B+1)
    return pval


### Experiments functions
def exp1(it, theta, gamma, c, a, b, skew, m, n, p, loss, alpha, B, 
         tests={'stfr':True, 'resit':True, 'gcm':True, 'crt':True, 'cpt':True,'rbpt':True, 'rbpt2':True}):
    
    #Gen. training data
    Z_train=sample_z(m, p, random_state=2*it)
    X_train=sample_x(Z_train, b, random_state=2*it)
    Y_train=sample_y(X_train, Z_train, a, b, c, gamma, skew, random_state=2*it)

    #Gen. test data
    Z_test=sample_z(n, p, random_state=2*it+1)
    X_test=sample_x(Z_test, b, random_state=2*it+1)
    Y_test=sample_y(X_test, Z_test, a, b, c, gamma, skew, random_state=2*it+1)  
            
    #Fitting models
    g1 = g()
    g1.fit(X_train, Z_train, Y_train)
    g2 = g()
    g2.fit(None, Z_train, Y_train)
    g3 = g()
    g3.fit(None, Z_train, X_train)
                
    #STFR
    if tests['stfr']: 
        start_time = time.time()
        reject_stfr = (get_pval_stfr(X_test, Z_test, Y_test, g1, g2, loss=loss) <= alpha)
        time_stfr = time.time() - start_time
    else: 
        reject_stfr = np.nan
        time_stfr = np.nan
    
        
    #RESIT
    if tests['resit']: 
        start_time = time.time()
        reject_resit = (get_pval_resit(X_test, Z_test, Y_test, g2, g3, B=B) <= alpha)
        time_resit = time.time() - start_time
    else: 
        reject_resit = np.nan
        time_resit = np.nan
    
    
    #GCM
    if tests['gcm']: 
        start_time = time.time()
        reject_gcm = (get_pval_gcm(X_test, Z_test, Y_test, g2, g3) <= alpha)
        time_gcm = time.time() - start_time
    else: 
        reject_gcm = np.nan
        time_gcm = np.nan
    
    
    #CRT
    if tests['crt']: 
        start_time = time.time()
        reject_crt = (get_pval_crt(X_test, Z_test, Y_test, b, g1, g2, theta, B, loss=loss) <= alpha)
        time_crt = time.time() - start_time
    else: 
        reject_crt = np.nan   
        time_crt = np.nan   
    
    
    #CPT
    if tests['cpt']: 
        start_time = time.time()
        reject_cpt = (get_pval_cpt(X_test, Z_test, Y_test, b, g1, g2, theta, B, loss=loss) <= alpha)
        time_cpt = time.time() - start_time
    else: 
        reject_cpt = np.nan
        time_cpt = np.nan
    
    
    #RBPT
    if tests['rbpt']: 
        start_time = time.time()
        reject_rbpt = (get_pval_rbpt(X_test, Z_test, Y_test, b, g1, theta, loss=loss) <= alpha)
        time_rbpt = time.time() - start_time
    else: 
        reject_rbpt = np.nan
        time_rbpt = np.nan
    
    
    #RBPT2
    if tests['rbpt2']: 
        start_time = time.time()
        k=10
        h = GridSearchCV(KernelRidge(kernel='poly'), cv=2, n_jobs=1, scoring='neg_mean_squared_error',
                         param_grid={"alpha": np.logspace(0,-k,k), "degree": [2]}) 
         
        #import sklearn
        #d = sklearn.metrics.pairwise_distances(Z_train)
        #d = np.median(d)
        #h = GridSearchCV(KernelRidge(kernel='rbf'), cv=2, n_jobs=1, scoring='neg_mean_squared_error',
        #                 param_grid={"alpha":np.logspace(0,-k,k), "gamma": [.5*(1/d**2)]}) 
        
        #h = GridSearchCV(DecisionTreeRegressor(random_state=0), cv=2, n_jobs=1, scoring='neg_mean_squared_error',
        #                 param_grid={"min_samples_leaf": [5,10,25,50]})
        #h = GridSearchCV(KNeighborsRegressor(), cv=2, n_jobs=1, scoring='neg_mean_squared_error',
        #                 param_grid={"n_neighbors": [5,10,25,50], 'weights':['distance']})
        #h = GridSearchCV(lgb.LGBMRegressor(random_state=0, linear_tree=True), cv=2, n_jobs=1, scoring='neg_mean_squared_error',
        #                 param_grid={'n_estimators': [50,100]})
        
        #model = Pipeline([('poly', PolynomialFeatures(degree=2)),
        #                  ('linear', Ridge())])
        
        #h = GridSearchCV(model, cv=2, n_jobs=1, scoring='neg_mean_squared_error',
        #                 param_grid={"linear__alpha": [1e2, 1e0, 1e-2, 1e-4]})
        
        #h = GridSearchCV(MLPRegressor(random_state=1, solver='lbfgs', hidden_layer_sizes=(20,), tol=0.001, activation='logistic'), cv=2, n_jobs=1,             scoring='neg_mean_squared_error',param_grid={"alpha": [1e2, 1e0, 1e-2]})
                         
        h.fit(Z_train, g1.predict(X_train,Z_train).squeeze())
        
        reject_rbpt2 = (get_pval_rbpt2(X_test, Z_test, Y_test, g1, h, loss=loss) <= alpha)
        time_rbpt2 = time.time() - start_time
    else: 
        reject_rbpt2 = np.nan
        time_rbpt2 = np.nan
    
    
    #Output
    return [reject_stfr, reject_resit, reject_gcm, reject_crt, reject_cpt, reject_rbpt, reject_rbpt2,
            time_stfr, time_resit, time_gcm, time_crt, time_cpt, time_rbpt, time_rbpt2] 


### Plots
def plot_type1(results, sett, skew, loss, alpha=.05, xlabel='x', ylabel='y', legend=True):
    ####
    colors=['#722ef9','#2F58EB','#2fadeb','#EB9846','#D9120B'] 
    if '1_1' in sett: names=['STFR','RESIT','GCM','RBPT', 'RBPT2'] 
    else: names=['CRT', 'CPT','RBPT']
        
    ####
    columns_by = ['set','c','theta','skew','gamma','loss']
    results_mean = results.groupby(by=columns_by).mean().reset_index()
    results_std = results.groupby(by=columns_by).std().reset_index()
    
    ####
    ind_row = (results_mean.set==sett)&(results_mean.loss==loss)&(results_mean['skew']==skew)
    ind_col=[s.lower() for s in names]
    if '1_1' in sett: x = np.array(results_mean.loc[ind_row].gamma)
    else: x = np.array(results_mean.loc[ind_row].theta)
    results_mean = np.array(results_mean.loc[ind_row,ind_col])
    results_std = np.array(results_std.loc[ind_row,ind_col])
    
    ####
    for j in range(len(names)):
        if '1_1' in sett: i = j
        else: i=j+1
        plt.plot(x, results_mean[:,j], label=names[j], color=colors[i], lw=1, alpha=1)
        plt.fill_between(x, results_mean[:,j]-results_std[:,j], results_mean[:,j]+results_std[:,j], color=colors[i], alpha=.15)
    
    if legend:
        if '1_1' in sett: plt.legend(bbox_to_anchor=(.625, .65), loc='upper left', ncol = 1, prop={'size': 9.25}, borderaxespad=.0, frameon=True, framealpha=.6)                
        else: plt.legend(bbox_to_anchor=(.0225, .97), loc='upper left', ncol = 1, prop={'size': 10}, borderaxespad=.0, frameon=True, framealpha=.6)                
    plt.axhline(y=alpha, color='r', linestyle='--', lw=1, alpha=.5)
    plt.xticks(fontsize=11)
    plt.xlabel(xlabel, size=13)

    plt.yticks(fontsize=11)
    plt.ylabel(ylabel, size=13)
    plt.ylim(0,1)
    #plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%1.2f'))
    plt.grid(alpha=.2)
    
def plot_power(results, skew, loss, alpha=.05, xlabel='x', ylabel='y', legend=True):
    
    ####
    sett='power'
    colors=['#2fadeb','#EB9846','#D9120B'] 
    names=['CPT','RBPT', 'RBPT2']
    columns_by = ['set','c','theta','skew','gamma','loss']
    results_mean = results.groupby(by=columns_by).mean().reset_index()
    results_std = results.groupby(by=columns_by).std().reset_index()
    ind_row = (results_mean.set==sett)&(results_mean.loss==loss)&(results_mean['skew']==skew)
    ind_col=[s.lower() for s in names]
    x = np.array(results_mean.loc[ind_row].c)
    results_mean = np.array(results_mean.loc[ind_row,ind_col])
    results_std = np.array(results_std.loc[ind_row,ind_col])
    for j in range(len(names)):
        plt.plot(x, results_mean[:,j], label=names[j], color=colors[j], lw=1, alpha=1)
        plt.fill_between(x, results_mean[:,j]-results_std[:,j], results_mean[:,j]+results_std[:,j], color=colors[j], alpha=.15)
    
    if legend:
        plt.legend(bbox_to_anchor=(.0225, .97), loc='upper left', ncol = 1, prop={'size': 10}, borderaxespad=.0, frameon=True, framealpha=.6)                
    plt.axhline(y=alpha, color='r', linestyle='--', lw=1, alpha=.5)
    plt.xticks(fontsize=11)
    plt.xlabel(xlabel, size=13)
    plt.yticks(fontsize=11)
    plt.ylabel(ylabel, size=13)
    plt.ylim(0,1)
    #plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%1.2f'))
    plt.grid(alpha=.2)