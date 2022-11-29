from sklearn.metrics import r2_score 
import pandas as pd
import numpy as np
import math as m
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as mp

data = pd.read_csv('NYC_Bicycle_Counts_2016_Corrected.csv')
dataset_1 = pd.read_csv('NYC_Bicycle_Counts_2016_Corrected.csv')
dataset_1['Brooklyn Bridge']      = pd.to_numeric(dataset_1['Brooklyn Bridge'].replace(',','', regex=True))
dataset_1['Manhattan Bridge']     = pd.to_numeric(dataset_1['Manhattan Bridge'].replace(',','', regex=True))
dataset_1['Queensboro Bridge']    = pd.to_numeric(dataset_1['Queensboro Bridge'].replace(',','', regex=True))
dataset_1['Williamsburg Bridge']  = pd.to_numeric(dataset_1['Williamsburg Bridge'].replace(',','', regex=True))
dataset_1['Total']                = pd.to_numeric(dataset_1['Total'].replace(',','', regex=True))

name = input("what is your name? ")
print(name)