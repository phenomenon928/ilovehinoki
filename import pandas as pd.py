import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

def main():

    print('----Problem 1----')
    first_bridge()
    print('----Problem 2----')
    second_weather()
    print('----Problem 3----')
    third_precip()

    return

# Path 1 Problem 1

def first_bridge():

    avg_list = [0]*4
    for x in range(0,4):
        list = bike_np[:,x+5]
        average = averageBridge(list)
        avg_list[x] = average
    bridge_list = ['Brooklyn Bridge', 'Manhattan Bridge', 'Williamsburg Bridge', 'Queensboro Bridge']
    three_bridge = selectBridge(avg_list,bridge_list)
    print('Selected Bridges:', ', '.join(three_bridge))

    print(list)

    return

def averageBridge(list):

    for i in range(0, len(list)):
        list[i] = list[i].replace(',','')
        list[i] = int(list[i])
    avg_list = sum(list) / len(list)

    return avg_list

def selectBridge(avg_list, bridge_list):

    min_avg = min(avg_list)
    index = avg_list.index(min_avg)
    avg_list.remove(min_avg)
    bridge_list.remove(bridge_list[index])

    return bridge_list

# Path 1 Problem 2

def second_weather():

    high_list = bike_np[:,2]
    low_list = bike_np[:,3]
    avg_list = averageTemp(high_list, low_list)
    total_list = bike_np[:,9]
    for i in range(0, len(total_list)):
        total_list[i] = total_list[i].replace(',','')
        total_list[i] = int(total_list[i])
    r_sq, inter, coef, y_pred = tempRegression(avg_list, total_list)
    print('Linear Model: y = %.4f x + %.4f' %(coef[0], inter))
    print('R square: %.4f' %(r_sq))
    tempPlot(avg_list, total_list, y_pred)

    return

def averageTemp(high_temp, low_temp):

    avg_temp = [0] * len(high_temp)
    for i in range(len(high_temp)):
        avg_temp[i] = (high_temp[i] + low_temp[i]) / 2

    return avg_temp

def tempRegression(temp, total):

    x = np.array(temp).reshape((-1,1))
    y = np.array(total)
    model = LinearRegression()
    model.fit(x,y)
    r_sq = model.score(x,y)
    inter = model.intercept_
    coef = model.coef_
    y_pred = model.predict(x)

    return r_sq, inter, coef, y_pred

def tempPlot(avg_temp, total, total_pred):

    plt.scatter(avg_temp, total)
    plt.plot(avg_temp, total_pred, 'r')
    plt.xlabel('Average Temperature')
    plt.ylabel('Total Number of Bicyclists')
    plt.title('Predict Number of Bicyclists Depending on Weather')
    plt.show()

    return

# Path 1 Problem 3

def third_precip():

    precip_list = bike_np[:,4]
    for i in range(0, len(precip_list)):
        if precip_list[i] == 'T':
            precip_list[i] = 0
        else:
            if any(c.isalpha() for c in precip_list[i]) == False:
                precip_list[i] = float(precip_list[i])
            else:
                precip_list[i] = 0
    tot_list = bike_np[:,9]
    r_sq, inter, coef, y_pred = precipRegression(precip_list, tot_list)
    print('Linear Model: y = %.4f x + %.4f' % (coef[0], inter))
    print('R square: %.4f' % (r_sq))
    rainPlot(precip_list, tot_list, y_pred)

    return

def precipRegression(rain,tot):

    x = np.array(rain).reshape((-1, 1))
    y = np.array(tot)
    model = LinearRegression()
    model.fit(x, y)
    r_sq = model.score(x, y)
    inter = model.intercept_
    coef = model.coef_
    y_pred = model.predict(x)

    return r_sq, inter, coef, y_pred

def rainPlot(pre, tot, pred_tot):

    plt.scatter(pre, tot)
    plt.plot(pre, pred_tot, 'r')
    plt.xlabel('Precipitation')
    plt.ylabel('Total Number of Bicyclists')
    plt.title('Predict Number of Bicyclists Depending on Precipitation')
    plt.show()

    return

if __name__ == '__main__':

    bike = pd.read_csv('NYC_Bicycle_Counts_2016_Corrected.csv')
    bike_np = bike.values
    bike_traffic = main()
