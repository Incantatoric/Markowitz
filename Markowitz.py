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


# the contraint stating the sum of all the ratio should equal 1
def weight_sum_constraint(x):
    return x.sum() - 1.0


TrailingNum = 12
# test
Start = 0


def wf_analysis(Data, Start, TrailingNum):
    # store all the walk forward ROR data
    WfROR = []
    # store all the equal rate ROR
    EqROR = []

    while TrailingNum + Start < len(Data.index):
        Data1 = Data[Start:Start+TrailingNum]

        # average ROR, but in this case, we use the 1-year compound rate
        AvgRORList = (1 + Data1.iloc[:, 1:].mean()) ** 12 - 1

        # the objective function we want to minimize/maximize
        # given ratio, we calculate the portfolio's variance / sharp ratio etc
        def objective(x):
            mean = x @ pd.DataFrame(AvgRORList)
            variance = x.T @ Data1.cov() @ x
            sigma = variance ** 0.5
            # RFROR = 0.0125 / 12
            return -mean / sigma

        result = minimize(fun=objective,
                          x0=InitialValue,
                          method='SLSQP',
                          constraints={'type': 'eq', 'fun': weight_sum_constraint},
                          options={'ftol': 1e-20, 'maxiter': 800},
                          bounds=bnds)

        WfROR.append(Data.iloc[Start+TrailingNum, 1:].T @ result.x)
        EqROR.append(Data.iloc[Start+TrailingNum, 1:].T @ InitialValue)
        Start = Start + 1
    return WfROR, EqROR


WfROR, EqROR = wf_analysis(Data, Start, TrailingNum)
# end = time.time()
# print(end - start)

trace0 = go.Scatter(
    x=Data['날짜'][TrailingNum:],
    y=pd.Series(WfROR).cumsum()
)

trace1 = go.Scatter(
    x=Data['날짜'][TrailingNum:],
    y=pd.Series(EqROR).cumsum()
)

# plotly.offline.plot([trace0, trace1])
py.plot([trace0, trace1])



