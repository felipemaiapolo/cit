import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import skewnorm


def mu_x(Z, b):
    return (Z@b)**2

def sample_z(n,p):
    return np.random.normal(0, 1, n*p).reshape(n,p) 

def sample_x(Z, b):
    mu = mu_x(Z, b)
    return np.random.normal(mu,1)
   
def sample_y(X, Z, a, b, c, gamma, skew=0):   
    mu = c*X + Z@a + gamma*mu_x(Z, b)
    return np.array([skewnorm.rvs(skew, loc=m, scale=1, size=1).tolist() for m in mu]).reshape(-1,1)


def get_loss(y, y_hat, loss='mae'):
    
    assert y.shape==y_hat.shape
    
    if loss=='mae':
        return np.abs(y-y_hat)
    if loss=='mse':
        return (y-y_hat)**2
    
class g:
    def __init__(self):
        pass
    
    def fit(self, X, Z, Y):
        if X is None:  
            self.model = LinearRegression().fit(Z, Y)
        else:
            W=np.hstack((X, Z))
            self.model = LinearRegression().fit(W, Y)

    def predict(self, X, Z):
        if X is None:  
            return self.model.predict(Z)
        else:
            W=np.hstack((X, Z))
            return self.model.predict(W)

