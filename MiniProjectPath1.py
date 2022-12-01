import numpy as np
import pandas as pd
import statistics as st
import matplotlib.pyplot as mp
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

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
    test2 = Abs_Avg_Dev_best_bridges(dataset_1)
    print("Selected Bridges:", Select_best_bridges(test1, test2, bridge_name))
    
    print("")
    print("Problem 2:")
    user_input_1()
    n_total = temp(dataset_1)

    print("")
    print("Problem 3:")
    third_problem(n_total, dataset_1)

    print("")

    return 0

def data_preprocessing(array):
    arr = np.array(array)
    q3, q1 = np.percentile(arr, (75, 25))
    iqr = q3 - q1
    new_q3 = q3 + 1.5 * iqr
    new_q1 = q1 - 1.5 * iqr
    new_arr = arr
    new_arr = arr[(arr > new_q1) & (arr < new_q3)]

    return new_arr

def sort(list1, list2):
    sorted_list1, sorted_list2 = (list(t) for t in zip(*sorted(zip(list1, list2))))
    return sorted_list1, sorted_list2

# First Problem

def R_2_best_bridges(dataset_1):

    # Calculating top three bridges using R-squared method after preprocessing the data using IQR

    # before data prepreocessing
    # new_arr1 = dataset_1['Brooklyn Bridge']
    # new_arr2 = dataset_1['Manhattan Bridge']
    # new_arr3 = dataset_1['Queensboro Bridge']
    # new_arr4 = dataset_1['Williamsburg Bridge']

    # after data prepreocessing
    new_arr1 = data_preprocessing(dataset_1['Brooklyn Bridge'])
    new_arr2 = data_preprocessing(dataset_1['Manhattan Bridge'])
    new_arr3 = data_preprocessing(dataset_1['Queensboro Bridge'])
    new_arr4 = data_preprocessing(dataset_1['Williamsburg Bridge'])


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
    mp.scatter([x for x in range(len(new_arr1))], new_arr1 , color = 'c')
    mp.plot(x1, p(x1), 'r')

    for i in range(len(new_arr1)):
        pred1[i] = z[0] * a1[i] + z[1] 
    r2_list[0] = r2_score(new_arr1, pred1)
    fg1.text(0.17, 0.93, 'R-squared = %0.4f' % r2_list[0], horizontalalignment='center', verticalalignment='center', transform=fg1.transAxes, fontsize = 12, bbox=dict(facecolor='red', alpha=0.5))


    
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
    mp.scatter([x for x in range(len(new_arr2))], new_arr2, color = 'c')
    mp.plot(x2, p(x2), 'r')

    for i in range(len(new_arr2)):
        pred2[i] = z[0] * a2[i] + z[1] 
    r2_list[1] = r2_score(new_arr2, pred2)
    fg2.text(0.17, 0.93, 'R-squared = %0.4f' % r2_list[1], horizontalalignment='center', verticalalignment='center', transform=fg2.transAxes, fontsize = 12, bbox=dict(facecolor='red', alpha=0.5))


    
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
    mp.scatter([x for x in range(len(new_arr3))], new_arr3, color = 'c')
    mp.plot(x3, p(x3), 'r')

    for i in range(len(new_arr3)):
        pred3[i] = z[0] * a3[i] + z[1] 
    r2_list[2] = r2_score(new_arr3, pred3)
    fg3.text(0.17, 0.93, 'R-squared = %0.4f' % r2_list[2], horizontalalignment='center', verticalalignment='center', transform=fg3.transAxes, fontsize = 12, bbox=dict(facecolor='red', alpha=0.5))


    
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
    mp.scatter([x for x in range(len(new_arr4))], new_arr4, color = 'c')
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

def Abs_Avg_Dev_best_bridges(dataset_1):

    #Calculating top three bridges using absolute average deviation function

    Bridge_list = [0]*4
    Bridge_list[0] = sum(data_preprocessing((dataset_1['Brooklyn Bridge'])))
    Bridge_list[1] = sum(data_preprocessing(dataset_1['Manhattan Bridge']))
    Bridge_list[2] = sum(data_preprocessing(dataset_1['Williamsburg Bridge']))
    Bridge_list[3] = sum(data_preprocessing(dataset_1['Queensboro Bridge']))
    x_axis = ['Brooklyn', 'Manhattan', 'Williamsburg', 'Queensboro']

    y_mean = [np.mean(Bridge_list)]*len(Bridge_list)
    abs_avg_dev = [0]*4
    for i in range(len(y_mean)):
        abs_avg_dev[i] = abs(Bridge_list[i] - y_mean[i])

    mp.figure(2)
    mp.bar(x_axis, Bridge_list, color = 'c')
    mp.plot(x_axis,y_mean, label='Mean', linestyle='--', color = 'red')
    mp.xlabel('Name of Bridges')
    mp.ylabel('Total Number of Bicyclists (million)')
    mp.title('Total Number of Bicyclists for each Bridge', fontsize = 15)
    mp.legend()
    mp.show()

    largest_deviation = abs_avg_dev.index(max(abs_avg_dev))

    return(largest_deviation)

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

    # finds the average temperature in each day
    avg_temp = [0]*len(dataset_1)

    for i in range(len(avg_temp)):
        avg_temp[i] = (dataset_1['High Temp'][i] + dataset_1['Low Temp'][i]) / 2

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
    model1 = LinearRegression()
    model1.fit(x_1, y_1)
    # r_sq = model.score(x_1, y_1)
    # print(r_sq)
    # print("coefficient:", model.coef_)
    # print("y-intercent:", model.intercept_)
    mp.scatter(x_p, y_precipitation, label='Precipitation', color = 'c')
    mp.plot(x_1, model1.predict(x_1), label = 'Linear Regression', color = 'dodgerblue')
    mp.plot(x_p, p_p(x_p),label = 'Trendline', color = 'purple')
    mp.xlabel('Daily Average Temperature (°F)')
    mp.ylabel('Total Bike Traffic (24 hr period)')
    mp.legend()

    mp.subplot(3,1,2)
    x_2 = np.array(x_np).reshape((-1,1))
    y_2 = np.array(y_noprecipitation)
    model2 = LinearRegression()
    model2.fit(x_2, y_2)
    # print("coefficient:", model.coef_)
    # print("y-intercent:", model.intercept_)
    mp.scatter(x_np, y_noprecipitation, label='No Precipitation', color = 'coral')
    mp.plot(x_2, model2.predict(x_2), label = 'Linear Regression', color = 'red')
    mp.plot(x_np, p_np(x_np),label = 'Trendline', color = 'purple')
    mp.xlabel('Daily Average Temperature (°F)')
    mp.ylabel('Total Bike Traffic (24 hr period)')
    mp.legend()

    mp.subplot(3,1,3)
    mp.scatter(x_p, y_precipitation, label='Precipitation', color = 'c')
    mp.scatter(x_np, y_noprecipitation, label='No Precipitation', color = 'coral')
    mp.plot(x_1, model1.predict(x_1), label = 'Linear Regression (Precipitation)', color = 'dodgerblue')
    mp.plot(x_2, model2.predict(x_2), label = 'Linear Regression (No Precipitation)', color = 'red')
    mp.xlabel('Daily Average Temperature (°F)')
    mp.ylabel('Total Bike Traffic (24 hr period)')
    mp.legend()
    mp.show()

    return new_total

def user_input_1():
    pre = float(input("Enter Precipitation: "))
    high_temp = float(input('Enter Highest Temperature (°F): '))
    low_temp = float(input('Enter Lowest Temperature (°F): '))

    avg_temp = (high_temp + low_temp) / 2

    if pre > 0:
        total = (257.42705815 * avg_temp  - 5250.524153541815)
    else:
        total = (136.48628816 * avg_temp + 7713.267222284023)

    print("Predicted Traffic:", int(total))

    return 0

# Third Problem

def third_problem(n_total, dataset_1):
   
    day_dict = {'Monday': 0, 'Tuesday': 0, 'Wednesday': 0, 'Thursday': 0, 'Friday': 0, 'Saturday': 0, 'Sunday': 0}
    mon = 0
    tue = 0
    wed = 0
    thu = 0
    fri = 0
    sat = 0
    sun = 0
    
    for i in range(len(n_total)):
        day_dict[dataset_1['Day'][i]] += n_total[i]
        if dataset_1['Day'][i] == 'Monday':
            mon += 1
        if dataset_1['Day'][i] == 'Tuesday':
            tue += 1
        if dataset_1['Day'][i] == 'Wednesday':
            wed += 1
        if dataset_1['Day'][i] == 'Thursday':
            thu += 1
        if dataset_1['Day'][i] == 'Friday':
            fri += 1
        if dataset_1['Day'][i] == 'Saturday':
            sat += 1
        if dataset_1['Day'][i] == 'Sunday':
            sun += 1

    day_li = [mon, tue, wed, thu, fri, sat, sun]
    for i in range(7):
        day_dict[list(day_dict.keys())[i]] = day_dict[list(day_dict.keys())[i]] / day_li[i]
    
    mp.figure()
    myList = day_dict.items() 
    x, y = zip(*myList) 

    mp.plot(x, y)
    mp.title('Average Bicyclist for each Day')
    mp.xlabel('Day')
    mp.ylabel('Average Number of Bicyclist')
    mp.show()
    
    # user input for prediction
    pred_day = float(input("Enter the total number: "))
    pred_diff=[0]*7

    for i in range(len(pred_diff)):
        pred_diff[i] = abs(list(day_dict.values())[i] - pred_day)
    ind = pred_diff.index(min(pred_diff))
    print("Predicted Day:", list(day_dict.keys())[ind])

    return 0


if __name__ == '__main__':
    main()