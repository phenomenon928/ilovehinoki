import pandas as pd
import matplotlib.pyplot as mp
import numpy as np
from sklearn.linear_model import LinearRegression
import math as m
from sklearn.metrics import r2_score

''' 
The following is the starting code for path1 for data reading to make your first step easier.
'dataset_1' is the clean data for path1.
'''


# The `NYC_Bicycle_Counts_2016_Corrected.csv` gives information on bike traffic across a number of bridges in New York City. In this path, the analysis questions we would like you to answer are as follows:

# 1. You want to install sensors on the bridges to estimate overall traffic across all the bridges. But you only have enough budget to install sensors on three of the four bridges. Which bridges should you install the sensors on to get the best prediction of overall traffic?
# 2. The city administration is cracking down on helmet laws, and wants to deploy police officers on days with high traffic to hand out citations. Can they use the next day's weather forecast(low/high temperature and precipitation) to predict the total number of bicyclists that day? 
# 3. Can you use this data to predict what *day* (Monday to Sunday) is today based on the number of bicyclists on the bridges?

def main():
    best_bridges()

    return


dataset_1 = pd.read_csv('NYC_Bicycle_Counts_2016_Corrected.csv')
dataset_1['Brooklyn Bridge']      = pd.to_numeric(dataset_1['Brooklyn Bridge'].replace(',','', regex=True))
dataset_1['Manhattan Bridge']     = pd.to_numeric(dataset_1['Manhattan Bridge'].replace(',','', regex=True))
dataset_1['Queensboro Bridge']    = pd.to_numeric(dataset_1['Queensboro Bridge'].replace(',','', regex=True))
dataset_1['Williamsburg Bridge']  = pd.to_numeric(dataset_1['Williamsburg Bridge'].replace(',','', regex=True))
# print(dataset_1.to_string()) #This line will print out your data

def best_bridges():

    #Calculating top three bridges with the most traffic 

    bridge_list = ["Brooklyn Bridge", "Manhattan Bridge", "Williamsburg Bridge", "Queensboro Bridge"]


    # importing traffic data
    brooklyn_arr = data_np[:,5]
    brooklyn_list = brooklyn_arr.tolist()

    manhattan_arr = data_np[:,6]
    manhattan_list = manhattan_arr.tolist()

    williamsburg_arr = data_np[:,7]
    williamsburg = williamsburg_arr.tolist()

    queensboro_arr = data_np[:,8]
    queensboro_list = queensboro_arr.tolist()

    date_arr = data_np[:,0]
    date = date_arr.tolist()
    a = [0] * len(brooklyn_arr)
    count = 1
    for i in range(len(a)):
        a[i] += count
        count += 1
    
    coeff_brook = np.polyfit(a, brooklyn_list, 1)
    print(coeff_brook)

    # pred_brook = np.polyval(coeff_brook, date)
    # sse_brook = sum((brooklyn_list - coeff_brook) ** 2)
    # sst_brook = sum((brooklyn_list - m.mean(brooklyn_list)) ** 2)
    # r_brook = 1 - (sse_brook / sst_brook)
    # print(r_brook)

    # brook_matrix = np.corrcoef(date, brooklyn_list)
    # brook = brook_matrix[0,1]
    # R_sq = brook**2
    # print(R_sq)

    # R_square = r2_score(test_data, brooklyn_list) 
    # print('Coefficient of Determination', R_square) 




    # mp.scatter(date, brooklyn_list)
    # mp.plot(date, brooklyn_list, 'r')
    # mp.xlabel('days')
    # mp.ylabel('bike users')
    # mp.title("brooklyn")
    # mp.show()
  
    return

if __name__ == '__main__':
    data = pd.read_csv('NYC_Bicycle_Counts_2016_Corrected.csv')
    data_np = data.values
    data_traffic = main()
