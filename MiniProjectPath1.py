import pandas as pd
import matplotlib.pyplot as mp
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statistics as st

# 2. The city administration is cracking down on helmet laws, and wants to deploy police officers on days with high traffic to hand out citations. Can they use the next day's weather forecast(low/high temperature and precipitation) to predict the total number of bicyclists that day? 
# 3. Can you use this data to predict what *day* (Monday to Sunday) is today based on the number of bicyclists on the bridges?

def main():
    
    dataset_1 = pd.read_csv('NYC_Bicycle_Counts_2016_Corrected.csv')
    dataset_1['Brooklyn Bridge'] = pd.to_numeric(dataset_1['Brooklyn Bridge'].replace(',','', regex=True))
    dataset_1['Manhattan Bridge']     = pd.to_numeric(dataset_1['Manhattan Bridge'].replace(',','', regex=True))
    dataset_1['Queensboro Bridge']    = pd.to_numeric(dataset_1['Queensboro Bridge'].replace(',','', regex=True))
    dataset_1['Williamsburg Bridge']  = pd.to_numeric(dataset_1['Williamsburg Bridge'].replace(',','', regex=True))
    dataset_1['Total']                = pd.to_numeric(dataset_1['Total'].replace(',','', regex=True))
    bridge_name = ["Brooklyn Bridge", "Manhattan Bridge", "Williamsburg Bridge", "Queensboro Bridge"]

    print("")
    print("Problem 1:")
    test1 = R_2_best_bridges(dataset_1)
    test2 = Total_best_bridges(dataset_1)
    print("Selected Bridges:", Select_best_bridges(test1, test2, bridge_name))
    
    print("")
    print("Problem 2:")
    
    temp(dataset_1)

    print("")
    print("Problem 3:")


    print("")

    return 0

# First Problem

def R_2_best_bridges(dataset_1):

    #Calculating top three bridges using R-squared method
    r2_list = [0]*4
    a = [0]*len(dataset_1)
    count = 1
    pred = [0]*len(dataset_1)
    for i in range(len(a)):
        a[i] += count
        count += 1


    # R_2 plot for Brooklyn Bridge
    x = [x for x in range(len(dataset_1['Brooklyn Bridge']))]
    z = np.polyfit([x for x in range(len(dataset_1['Brooklyn Bridge']))], dataset_1["Brooklyn Bridge"],1)
    p = np.poly1d(z)
    
    #list of trendline(prediction) data
    for i in range(len(dataset_1)):
        pred[i] = z[0] * a[i] + z[1]
    trendline = r2_score(dataset_1["Brooklyn Bridge"], pred)
    r2_list[0] = trendline
    # print(trendline)

    fg1 = mp.subplot(2, 2, 1)
    fg1.text(0.17, 0.93, 'R-squared = %0.4f' % trendline, horizontalalignment='center', verticalalignment='center', transform=fg1.transAxes, fontsize = 12, bbox=dict(facecolor='red', alpha=0.5))
    mp.plot([x for x in range(len(dataset_1['Brooklyn Bridge']))], dataset_1["Brooklyn Bridge"], 'c')
    mp.plot(x, p(x), 'r')
    

    # R_2 plot for Manhattan Bridge
    x = [x for x in range(len(dataset_1['Manhattan Bridge']))]
    z = np.polyfit([x for x in range(len(dataset_1['Manhattan Bridge']))], dataset_1["Manhattan Bridge"],1)
    p = np.poly1d(z)

    for i in range(len(dataset_1)):
        pred[i] = z[0] * a[i] + z[1]
    trendline = r2_score(dataset_1["Manhattan Bridge"], pred)
    r2_list[1] = trendline
    # print(trendline)

    fg2 = mp.subplot(2, 2, 2)
    fg2.text(0.17, 0.93, 'R-squared = %0.4f' % trendline, horizontalalignment='center', verticalalignment='center', transform=fg2.transAxes, fontsize = 12, bbox=dict(facecolor='red', alpha=0.5))
    mp.plot([x for x in range(len(dataset_1['Manhattan Bridge']))], dataset_1["Manhattan Bridge"], 'c')
    mp.plot(x, p(x), 'r')


    # R_2 plot for Williamsburg Bridge
    x = [x for x in range(len(dataset_1['Williamsburg Bridge']))]

    z = np.polyfit([x for x in range(len(dataset_1['Williamsburg Bridge']))], dataset_1["Williamsburg Bridge"],1)
    p = np.poly1d(z)

    for i in range(len(dataset_1)):
        pred[i] = z[0] * a[i] + z[1]
    trendline = r2_score(dataset_1["Williamsburg Bridge"], pred)
    r2_list[2] = trendline
    # print(trendline)

    fg3 = mp.subplot(2, 2, 3)
    fg3.text(0.17, 0.93, 'R-squared = %0.4f' % trendline, horizontalalignment='center', verticalalignment='center', transform=fg3.transAxes, fontsize = 12, bbox=dict(facecolor='red', alpha=0.5))
    mp.plot([x for x in range(len(dataset_1['Williamsburg Bridge']))], dataset_1["Williamsburg Bridge"], 'c')
    mp.plot(x, p(x), 'r')


    # R_2 plot for Queensboro Bridge
    x = [x for x in range(len(dataset_1['Queensboro Bridge']))]

    z = np.polyfit([x for x in range(len(dataset_1['Queensboro Bridge']))], dataset_1["Queensboro Bridge"],1)
    p = np.poly1d(z)

    for i in range(len(dataset_1)):
        pred[i] = z[0] * a[i] + z[1]
    trendline = r2_score(dataset_1["Queensboro Bridge"], pred)
    r2_list[3] = trendline
    # print(trendline)

    fg4 = mp.subplot(2, 2, 4)
    fg4.text(0.17, 0.93, 'R-squared = %0.4f' % trendline, horizontalalignment='center', verticalalignment='center', transform=fg4.transAxes, fontsize = 12, bbox=dict(facecolor='red', alpha=0.5))
    mp.plot([x for x in range(len(dataset_1['Queensboro Bridge']))], dataset_1["Queensboro Bridge"], 'c')
    mp.plot(x, p(x), 'r')

    mp.suptitle('R-squared Plot for each Bridge', fontsize = 15)
    fg1.title.set_text("Brooklyn")
    fg1.set_ylabel('Traffic in each Day')
    fg1.set_xlabel("Day Since April 1st")
    fg2.title.set_text("Manhattan")
    fg2.set_ylabel('Traffic in each Day')
    fg2.set_xlabel("Day Since April 1st")
    fg3.title.set_text("Williamsburg")
    fg3.set_ylabel('Traffic in each Day')
    fg3.set_xlabel("Day Since April 1st")
    fg4.title.set_text("Brooklyn")
    fg4.set_ylabel('Traffic in each Day')
    fg4.set_xlabel("Day Since April 1st")
    mp.subplots_adjust(left = 0.1, bottom = 0.1, right = 0.9, top = 0.9, wspace = 0.3, hspace = 0.3)
    # mp.show()
    
    least_r2_index = r2_list.index(min(r2_list))
    return(least_r2_index)

def Total_best_bridges(dataset_1):

    #Calculating top three bridges with the most total traffic

    Bridge_list = [0]*4
    Bridge_list[0] = sum(dataset_1['Brooklyn Bridge'])
    Bridge_list[1] = sum(dataset_1['Manhattan Bridge'])
    Bridge_list[2] = sum(dataset_1['Williamsburg Bridge'])
    Bridge_list[3] = sum(dataset_1['Queensboro Bridge'])
    
    x_axis = ['Brooklyn', 'Manhattan', 'Williamsburg', 'Queensboro']

    mp.figure(2)
    mp.bar(x_axis, Bridge_list, color = 'c')
    mp.xlabel('Name of Bridges')
    mp.ylabel('Total Number of Bicyclists (million)')
    mp.title('Total Number of Bicyclists for each Bridge', fontsize = 15)
    # mp.show()

    least_total_index = Bridge_list.index(min(Bridge_list))

    return(least_total_index)

def Select_best_bridges(test1, test2, bridge_name):
    # finds and removes the least favorable bridges from the bridge_name
    
    # least favorable bridge indexs
    indexs = [test1, test2]
    index = st.mode(indexs)
    # removes least favorable bridge from the birdge least
    bridge_name.pop(index)

    return(bridge_name)
    
    
# Second Problem

def temp(dataset_1):
    avg_temp = [0]*len(dataset_1['High Temp'])

    for i in range(len(avg_temp)):
        avg_temp[i] = (dataset_1['High Temp'][i] + dataset_1['Low Temp'][i]) / 2

    print(avg_temp)

# Third Problem

if __name__ == '__main__':
    main()