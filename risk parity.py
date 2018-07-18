# the usual stuff
import os
import sys
import pandas as pd
import numpy as np

# our main tool for optimization
from scipy.optimize import minimize

# plotting
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
plotly.tools.set_credentials_file(username='Incantator', api_key='plAiR5R27boC650ZYo7R')
plotly.tools.set_config_file(world_readable=True)

# start = time.time()
os.chdir(r'C:\Users\quantec\Desktop\work at office\asset allocation\Markowitz')

# I wanna see each and every column rather than giving me the abbreviations '...'
pd.set_option('expand_frame_repr', False)

# pandas display floating precision to 10; default is 6
# https://pandas.pydata.org/pandas-docs/stable/options.html
pd.reset_option('display.precision', 10)

# data.columns = Index(['날짜', 'Coli'])
# the first column is the date, and the rest should be ROR data of the corresponding column name
Data = pd.read_excel(r'20170817 Markowiz Mean Variance Efficient Frontier_YS.xlsm', 'Sheet1')
Data['날짜'] = pd.to_datetime(Data['날짜'])

lb = [0, 0, 0, 0]
ub = [1, 1, 1, 1]
bnds = tuple(zip(lb, ub))

# initial ratio  ex) (0.25, 0.25, 0.25, 0.25)
InitialValue = np.repeat(1 / (len(Data.columns)-1), len(Data.columns)-1)

assert len(lb) == len(InitialValue), 'The length of the lower/upper bound and the initial ratio should be the same.'

# df = np.log(df)
# df = (df-df.shift(1)).iloc[1:]
# covmat = ((df-df.mean()).T @ (df-df.mean())) / (len(df)-1)

covmat = Data.iloc[:, 1:].cov()


def RiskParity(x):
    risk_diff = 0.00001 - (x.T @ covmat @ x) / (len(x) * covmat @ x)
    return (risk_diff ** 2).sum()


# 제약
def TargetVol_const_lower(x):
    variance = x.T @ covmat @ x
    sigma = variance ** 0.5
    sigma_scale = sigma * np.sqrt(12)

    vol_diffs = sigma_scale - (0.06 * 0.95)
    return (vol_diffs)


def TargetVol_const_upper(x):
    variance = x.T @ covmat @ x
    sigma = variance ** 0.5
    sigma_scale = sigma * np.sqrt(12)

    vol_diffs = (0.06 * 1.05) - sigma_scale
    return (vol_diffs)


def weight_con(x):
    return (sum(x)-1)

def con1(x):
    variance = x.T @ covmat * x
    return (variance[0] - variance[1])

def con2(x):
    variance = x.T @ covmat * x
    return (variance[1] - variance[2])

def con3(x):
    variance = x.T @ covmat * x
    return (variance[2] - variance[3])

# 타겟 변동성 설정 시 TargetVol_const_lower 함수와 TargetVol_const_upper 함수를 제약 함수로 활용할 것(비중 합계 100%로 만들기 어렵다는 문제 발생)
# 타겟 변동성 설정 없이 단순 동일 위험 기여도로 설정할 경우 con1,con2,con3,weight_con(비중 합계 100% 고정)을 제약 함수로 활용할 것
def minvar(covmat, bnds, InitialVAlue):
    constraints = ({'type': 'eq', 'fun': con1}, {'type': 'eq', 'fun': con2}, {'type': 'eq', 'fun': con3}, {'type': 'eq', 'fun': weight_con})

    options = {'ftol': 1e-20, 'maxiter': 80}

    result = minimize(fun=RiskParity,
                      x0=InitialValue,
                      method='SLSQP',
                      constraints=constraints,
                      options=options,
                      bounds=bnds)
    return (result.x)


v = minvar(covmat, bnds, InitialValue)
A = covmat @ v * v
sum(v)