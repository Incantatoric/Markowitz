from pandas import Series, DataFrame
import pandas as pd
from pandas.tseries.offsets import Day, MonthEnd
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from scipy.stats import stats
from scipy.optimize import minimize
import copy
import os

os.chdir(r'C:\Users\Administrator\Desktop\quantec_강상일\자산배분\전략 적용')
pd.set_option('expand_frame_repr', False)


data = pd.read_excel("total.xlsx", sheet_name='Sheet1')
data = data.sort_values(by=['date'], axis=0)
kospi = pd.read_excel("kospi_daily.xlsx", sheet_name='Sheet1').sort_values(by=['date'], axis=0)
kospi['date'] = kospi['date'].dt.strftime('%Y%m')
kospi = kospi.groupby(['date']).apply(lambda x: x.iloc[-1])
date = pd.DataFrame(data['date'].dt.strftime('%Y%m'))
temp = pd.merge(date, kospi, how='inner', on='date')
kospi = temp.append(kospi[kospi['date']=='201804']).sort_values(by=['date'], axis=0).reset_index(drop=['index'])
kospi['return'] = kospi['close'].shift(-1)/kospi['close']-1
kospi = kospi[~pd.isnull(kospi).any(axis=1)]



df = copy.deepcopy(data.iloc[:,3:])

for i in df.columns:
    df['%s'%i] = df['%s'%i] - kospi['return']


result = df.corr()

writer = pd.ExcelWriter('전략간상관계수.xlsx', engine = 'xlsxwriter')
result.to_excel(writer, sheet_name='Sheet1')
writer.save()
writer.close()
plt.matshow(df.corr())

result2 = df.cov()

writer = pd.ExcelWriter('전략간공분산.xlsx', engine = 'xlsxwriter')
result2.to_excel(writer, sheet_name='Sheet1')
writer.save()
writer.close()
plt.matshow(df.cov())