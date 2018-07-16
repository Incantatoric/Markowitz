import os, sys
import pandas as pd
from scipy.optimize import minimize
import numpy as np

os.chdir(r'C:\Users\quantec\Desktop\work at office\asset allocation')

pd.set_option('expand_frame_repr', False)

# data.columns = Index(['날짜', 'Coli']) where each column (other than '날짜') contains its ROR
data = pd.read_excel(r'20170817 Markowiz Mean Variance Efficient Frontier_YS.xlsm', 'Sheet1')
data['날짜'] = pd.to_datetime(data['날짜'])

# average ROR; here, we're converting monthly ROR into yearly one as a compound rate
AvgRORList = []
for i in range(1, len(data.columns)):
    AvgRORList.append((1 + data.iloc[:, i].mean())**12 - 1)

# covariance ndarray
CovMat = np.cov((data['SPY'], data['GLD'], data['UUP'], data['IWM']))


# only minimize is possible, so in case of maximizing, we apply the minus sign
def objective(x):
    mean = x @ pd.DataFrame(AvgRORList)
    variance = x.T @ covmat @ x
    sigma = variance ** 0.5
    # RFROR = 0.0125 / 12
    return -mean / sigma


def weight_sum_constraint(x):
        return x.sum() - 1.0


lb = [0.1, 0, 0.15, 0]
ub = [1, 0.9, 1, 0.9]


def markowitz_ratio(covmat, lb, ub):
    # initial value
    x0 = np.repeat(1 / covmat.shape[0], covmat.shape[0])
    bnds = tuple(zip(lb, ub))

    constraints = ({'type': 'eq', 'fun': weight_sum_constraint})
    options = {'ftol': 1e-20, 'maxiter': 800}

    result = minimize(fun=MinVol_objective,
                      x0=x0,
                      method='SLSQP',
                      constraints=constraints,
                      options=options,
                      bounds=bnds)
    return result.x


markowitz_ratio(CovMat, lb, ub)

