from sklearn.metrics import r2_score 
import pandas as pd
import numpy as np
import math as m

data = pd.read_csv('NYC_Bicycle_Counts_2016_Corrected.csv')
data_np = data.values
dataaa = data_np[:,5]
# print(data.to_string())


a = [0] * len(dataaa)
count = 1
for i in range(len(a)):
    a[i] += count
    count += 1

dataaa = [int(i) for i in dataaa]
# dataaa = dataaa.replace(',','', regex=True)
print(dataaa)
# R_square = r2_score(a, dataaa) 
# print('Coefficient of Determination', R_square)


# coeff_brook = np.polyfit(a, dataaa, 1, rcond=None, full=False, w=None, cov=False)
# pred_brook = np.polyval(coeff_brook, a)
# sse_brook = sum((dataaa - coeff_brook) ** 2)
# sst_brook = sum((dataaa - m.mean(dataaa)) ** 2)
# r_brook = 1 - (sse_brook / sst_brook)
# print(r_brook)

# coeff_brook = np.polyfit(a, dataaa, 1)
# print(coeff_brook)