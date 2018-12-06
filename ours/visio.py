# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt

def visio(df):
    anomaly = df[df['label']==1]
    normal = df[df['label']==0]
    ax1 = anomaly.plot.scatter(x='timestamp', y='value', color='b', marker='s', label='anomaly') 
    ax2 = normal.plot.scatter(x='timestamp', y='value', color='g', alpha=0.2, label='normal', ax=ax1)
    plt.xlabel('timestamp', fontsize = 20)
    plt.ylabel('value', fontsize = 20)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.show()
def visio_line(df):
    ax1 = df.plot(x='timestamp', y='value', color='g')
    anomaly = df[df['label']==1]
    anomaly.plot.scatter(x='timestamp', y='value', color='b', marker='s', label='anomaly', ax = ax1) 
    plt.xlabel('timestamp', fontsize = 20)
    plt.ylabel('value', fontsize = 20)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.show()
def visio_ts(ts, timeseries):
    start = ts.start_stamp
    end = ts.start_stamp + ts.span*len(ts.ts)
    mark = []
    i = 0
    for stamp in range(start, end, ts.span):
        if stamp == timeseries[i]:
            mark.append(1)
            i += 1
        else:
            mark.append(0)
    df = pd.DataFrame()
    df['mark'] = pd.Series(mark)
    df['timestamp'] = pd.Series([stamp for stamp in range(start, end, ts.span)])
    df['value'] =  pd.Series(ts.get_ts())
    formal = df[df['mark'] == 1]
    fill = df[df['mark'] == 0]
    ax = formal.plot.scatter(x='timestamp', y='value', color='g', alpha=0.1, label='value')
    fill.plot.scatter(x='timestamp', y='value', color='m', alpha=0.1, marker = '*',label='filled value', ax = ax)
    plt.xlabel('timestamp', fontsize = 20)
    plt.ylabel('value', fontsize = 20)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.show()
    
def plot_metric_label(df, metric = 'c_a'):
    df= df[df[metric]<=10.0]
    anomaly = df[df['label']==1]
    normal = df[df['label']==0]
    ax = anomaly.plot.scatter(x='timestamp', y=metric, color='b', marker ='D',label='anomaly') #先设定第一个散点图，颜色为深蓝色标签为Group 1，以ab两列作为x及y轴的值 
    normal.plot.scatter(x='timestamp', y=metric, color='g', alpha=0.4, label='normal', ax=ax)
    plt.show()

def truth_predict(real, predict, delay = 7):
    counter = 0#记录异常片段的长度
    find = False
    ans = []
    for i in range(len(predict)):
        if real[i] == 0:#正常数据
            if find:
                for j in range(counter):
                    ans.append(1)#tp
                counter = 0
                find = False
            else:
                for j in range(counter):
                    ans.append(2)#fn
                counter = 0              
            if predict[i] == 0:
                ans.append(3)#tn
            else:
                ans.append(4)#fp
        else:
            if predict[i] != 0 and counter <= delay:
                find = True
            counter += 1
            #print counter
    if counter > 0 and find:#处理尾部是异常的情况
        for j in range(counter):
            ans.append(1)#tp
    if counter >0 and find == False:
        for j in range(counter):
            ans.append(3)#fn
    return ans
def plot_ans(df, ratio = 3):
    df = df[:len(df)/ratio]#容易导致后续四个dataframe为空而报错
    tp = df[df['y'] == 1]
    fn = df[df['y'] == 2]
    tn = df[df['y'] == 3]
    fp = df[df['y'] == 4]
    #print fn.describe()
    #print fp.describe()
    if len(fn) >0 and len(fp) >0:
        ax = tp.plot.scatter(x='timestamp', y='value', color='b', label='tp') 
        ax1 = tn.plot.scatter(x='timestamp',y = 'value', color = 'g',alpha = 0.15, ax =ax, label='tn')  
        ax2= fn.plot.scatter(x='timestamp', y='value', color='b',marker='x', label='fn', ax=ax1)
        fp.plot.scatter(x='timestamp',y = 'value', color = 'r', marker = 'x',ax =ax2, label='fp')
        plt.show()
    elif len(fn) > 0: #fp = 0
        ax = tp.plot.scatter(x='timestamp', y='value', color='b', label='tp') 
        ax1 = tn.plot.scatter(x='timestamp',y = 'value', color = 'g',alpha = 0.15, ax =ax, label='tn')  
        fn.plot.scatter(x='timestamp', y='value', color='b',marker='x', label='fn', ax=ax1)
        plt.show()
    elif len(fp) >0: #fn=0
        ax = tp.plot.scatter(x='timestamp', y='value', color='b', label='tp') 
        ax1 = tn.plot.scatter(x='timestamp',y = 'value', color = 'g',alpha = 0.15, ax =ax, label='tn')  
        fp.plot.scatter(x='timestamp',y = 'value', color = 'r', marker = 'x',ax =ax1, label='fp')
        plt.show()   
    else:
        ax = tp.plot.scatter(x='timestamp', y='value', color='b', label='tp') 
        tn.plot.scatter(x='timestamp',y = 'value', color = 'g',alpha = 0.15, ax =ax, label='tn')  
        plt.show()