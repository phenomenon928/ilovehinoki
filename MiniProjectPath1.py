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
    dataset_1['Brooklyn Bridge']      = pd.to_numeric(dataset_1['Brooklyn Bridge'].replace(',','', regex=True))
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
    # print(user_input())
    temp(dataset_1)

    print("")
    print("Problem 3:")


    print("")

    return 0

# First Problem
def data_preprocessing(array):
    arr = np.array(array)
    q3, q1 = np.percentile(arr, (75, 25))
    iqr = q3 - q1
    new_q3 = q3 + 1.5 * iqr
    new_q1 = q1 - 1.5 * iqr
    new_arr = arr
    new_arr = arr[(arr > new_q1) & (arr < new_q3)]

    return new_arr

def R_2_best_bridges(dataset_1):

    # Calculating top three bridges using R-squared method after preprocessing the data using IQR

    new_arr1 = data_preprocessing(dataset_1['Brooklyn Bridge'])
    a1 = [0]*len(new_arr1)
    count = 1
    for i in range(len(a1)):
            a1[i] += count
            count += 1

    r2_list = [0]*4

    x1 = [x for x in range(len(new_arr1))]
    pred1 = [0]*len(new_arr1)
    z = np.polyfit([x for x in range(len(new_arr1))], new_arr1,1)
    p = np.poly1d(z)

    fg1 = mp.subplot(2, 2, 1)
    mp.plot([x for x in range(len(new_arr1))], new_arr1 , color = 'c')
    mp.plot(x1, p(x1), 'r')

    for i in range(len(new_arr1)):
        pred1[i] = z[0] * a1[i] + z[1] 
    r2_list[0] = r2_score(new_arr1, pred1)
    fg1.text(0.17, 0.93, 'R-squared = %0.4f' % r2_list[0], horizontalalignment='center', verticalalignment='center', transform=fg1.transAxes, fontsize = 12, bbox=dict(facecolor='red', alpha=0.5))


    new_arr2 = data_preprocessing(dataset_1['Manhattan Bridge'])
    a2 = [0]*len(new_arr2)
    count = 1
    for i in range(len(a2)):
            a2[i] += count
            count += 1

    x2 = [x for x in range(len(new_arr2))]
    pred2 = [0]*len(new_arr2)
    z = np.polyfit([x for x in range(len(new_arr2))], new_arr2,1)
    p = np.poly1d(z)
    fg2 = mp.subplot(2, 2, 2)
    mp.plot([x for x in range(len(new_arr2))], new_arr2, color = 'c')
    mp.plot(x2, p(x2), 'r')

    for i in range(len(new_arr2)):
        pred2[i] = z[0] * a2[i] + z[1] 
    r2_list[1] = r2_score(new_arr2, pred2)
    fg2.text(0.17, 0.93, 'R-squared = %0.4f' % r2_list[1], horizontalalignment='center', verticalalignment='center', transform=fg2.transAxes, fontsize = 12, bbox=dict(facecolor='red', alpha=0.5))


    new_arr3 = data_preprocessing(dataset_1['Queensboro Bridge'])
    a3 = [0]*len(new_arr3)
    count = 1
    for i in range(len(a3)):
            a3[i] += count
            count += 1

    x3 = [x for x in range(len(new_arr3))]
    pred3 = [0]*len(new_arr3)
    z = np.polyfit([x for x in range(len(new_arr3))], new_arr3,1)
    p = np.poly1d(z)
    fg3 = mp.subplot(2, 2, 3)
    mp.plot([x for x in range(len(new_arr3))], new_arr3, color = 'c')
    mp.plot(x3, p(x3), 'r')

    for i in range(len(new_arr3)):
        pred3[i] = z[0] * a3[i] + z[1] 
    r2_list[2] = r2_score(new_arr3, pred3)
    fg3.text(0.17, 0.93, 'R-squared = %0.4f' % r2_list[2], horizontalalignment='center', verticalalignment='center', transform=fg3.transAxes, fontsize = 12, bbox=dict(facecolor='red', alpha=0.5))


    new_arr4 = data_preprocessing(dataset_1['Williamsburg Bridge'])
    a4 = [0]*len(new_arr4)
    count = 1
    for i in range(len(a4)):
            a4[i] += count
            count += 1

    x4 = [x for x in range(len(new_arr4))]
    pred4 = [0]*len(new_arr4)

    z = np.polyfit([x for x in range(len(new_arr4))], new_arr4,1)
    p = np.poly1d(z)
    fg4 = mp.subplot(2, 2, 4)
    mp.plot([x for x in range(len(new_arr4))], new_arr4, color = 'c')
    mp.plot(x4, p(x4), 'r')

    for i in range(len(new_arr4)):
        pred4[i] = z[0] * a4[i] + z[1] 
    r2_list[3] = r2_score(new_arr4, pred4)
    fg4.text(0.17, 0.93, 'R-squared = %0.4f' % r2_list[3], horizontalalignment='center', verticalalignment='center', transform=fg4.transAxes, fontsize = 12, bbox=dict(facecolor='red', alpha=0.5))

    mp.suptitle('R-squared Plot for each Bridge', fontsize = 15)
    fg1.title.set_text("Brooklyn")
    fg1.set_ylabel('Traffic in each Day')
    fg1.set_xlabel("Day Since April 1st")
    fg2.title.set_text("Manhattan")
    fg2.set_ylabel('Traffic in each Day')
    fg2.set_xlabel("Day Since April 1st")
    fg3.title.set_text("Quuensboro")
    fg3.set_ylabel('Traffic in each Day')
    fg3.set_xlabel("Day Since April 1st")
    fg4.title.set_text("Williamsburg")
    fg4.set_ylabel('Traffic in each Day')
    fg4.set_xlabel("Day Since April 1st")
    mp.subplots_adjust(left = 0.1, bottom = 0.1, right = 0.9, top = 0.9, wspace = 0.3, hspace = 0.3)
    mp.show()
    
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
    mp.show()

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
def sort(list1, list2):
    sorted_list1, sorted_list2 = (list(t) for t in zip(*sorted(zip(list1, list2))))
    return sorted_list1, sorted_list2

def temp(dataset_1):

    # # high temperature IQR

    # # low temperature IQR

    # finds the average temperature in each day
    avg_temp = [0]*len(dataset_1)

    for i in range(len(avg_temp)):
        avg_temp[i] = (dataset_1['High Temp'][i] + dataset_1['Low Temp'][i]) / 2 #여기 고쳐아함!!!!!


    # gets new total list excluding Brooklyn bridge
    new_total = [0]*len(dataset_1)

    for i in range(len(dataset_1)):
        new_total[i] = dataset_1['Total'][i] - dataset_1['Brooklyn Bridge'][i]
    
    # gets new list of indexes percipiation above 0 and at 0
    x_p = []
    x_np = []
    y_precipitation = []
    y_noprecipitation = []

    for i in range(len(dataset_1)):
        if dataset_1["Precipitation"][i] > 0:
            y_precipitation.append(new_total[i])
            x_p.append(avg_temp[i])
        else:
            y_noprecipitation.append(new_total[i])
            x_np.append(avg_temp[i])
    
    #sorting data
    x_p, y_precipitation = sort(x_p, y_precipitation)
    x_np, y_noprecipitation = sort(x_np, y_noprecipitation)
    
    # # trendlines
    z_p = np.polyfit(x_p, y_precipitation,2)
    p_p = np.poly1d(z_p)
    z_np = np.polyfit(x_np, y_noprecipitation,2)
    p_np = np.poly1d(z_np)

    mp.figure()
    mp.suptitle('Impact of Weather Condition on Bike Traffic', fontsize = 15)
    mp.subplots_adjust(left = 0.1, bottom = 0.1, right = 0.9, top = 0.9, wspace = 0.3, hspace = 0.3)

    mp.subplot(3,1,1)
    x_1 = np.array(x_p).reshape((-1,1))
    y_1 = np.array(y_precipitation)
    model = LinearRegression()
    model.fit(x_1, y_1)
    # r_sq = model.score(x_1, y_1)
    # print(r_sq)
    # print("coefficient:", model.coef_)
    # print("y-intercent:", model.intercept_)
    mp.scatter(x_p, y_precipitation, label='Precipitation')
    mp.plot(x_1, model.predict(x_1), label = 'Linear Regression', color = 'green')
    mp.plot(x_p, p_p(x_p),label = 'Trendline', color = 'r')
    mp.xlabel('Daily Average Temperature (°F)')
    mp.ylabel('Total Bike Traffic (24 hr period)')
    mp.legend()

    mp.subplot(3,1,2)
    x_2 = np.array(x_np).reshape((-1,1))
    y_2 = np.array(y_noprecipitation)
    model = LinearRegression()
    model.fit(x_2, y_2)
    # print("coefficient:", model.coef_)
    # print("y-intercent:", model.intercept_)
    mp.scatter(x_np, y_noprecipitation, label='No Precipitation', color = 'darkorange')
    mp.plot(x_2, model.predict(x_2), label = 'Linear Regression', color = 'green')
    mp.plot(x_np, p_np(x_np),label = 'Trendline', color = 'r')
    mp.xlabel('Daily Average Temperature (°F)')
    mp.ylabel('Total Bike Traffic (24 hr period)')
    mp.legend()

    mp.subplot(3,1,3)
    mp.scatter(x_p, y_precipitation, label='Precipitation')
    mp.scatter(x_np, y_noprecipitation, label='No Precipitation', color = 'darkorange')
    mp.xlabel('Daily Average Temperature (°F)')
    mp.ylabel('Total Bike Traffic (24 hr period)')
    mp.legend()
    mp.show()
    
    return 0

def user_input():
    pre = float(input("Enter Precipitation: "))
    high_temp = float(input('Enter Highest Temperature: '))
    low_temp = float(input('Enter Lowest Temperature: '))

    avg_temp = (high_temp + low_temp) / 2

    if pre > 0:
        total = (257.42705815 * avg_temp  - 5250.524153541815)
    else:
        total = (136.48628816 * avg_temp + 7713.267222284023)

    return total

# Third Problem

if __name__ == '__main__':
    main()