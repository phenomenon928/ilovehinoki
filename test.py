from sklearn.metrics import r2_score 
import pandas as pd
import numpy as np
import math as m
from sklearn.linear_model import LinearRegression

data = pd.read_csv('NYC_Bicycle_Counts_2016_Corrected.csv')
dataset_1 = pd.read_csv('NYC_Bicycle_Counts_2016_Corrected.csv')
dataset_1['Brooklyn Bridge']      = pd.to_numeric(dataset_1['Brooklyn Bridge'].replace(',','', regex=True))
dataset_1['Manhattan Bridge']     = pd.to_numeric(dataset_1['Manhattan Bridge'].replace(',','', regex=True))
dataset_1['Queensboro Bridge']    = pd.to_numeric(dataset_1['Queensboro Bridge'].replace(',','', regex=True))
dataset_1['Williamsburg Bridge']  = pd.to_numeric(dataset_1['Williamsburg Bridge'].replace(',','', regex=True))
dataset_1['Total']                = pd.to_numeric(dataset_1['Total'].replace(',','', regex=True))

# print(data.to_string())

# R_square = r2_score(a, dataaa) 
# print('Coefficient of Determination', R_square)

#initiate linear regression model
model = LinearRegression()

#define predictor and response variables
X, y = [dataset_1['Brooklyn Bridge']], dataset_1.score

#fit regression model
model.fit(X, y)

#calculate R-squared of regression model
r_squared = model.score(X, y)

#view R-squared value
print(r_squared)