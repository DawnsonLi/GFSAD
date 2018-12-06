# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from features import *
'''
为了实现只对每段异常的前T个样本进行
'''
def extract_Tsamples(label, T=7):
    mark = []
    counter = 0
    for i in range(len(label)):
        if label[i] == 0:
            mark.append(0)
            counter = 0
        else:
            if counter < T:
                mark.append(1)#在T内的
            else:
                mark.append(2)#不再T内的，但确实是异常
            counter += 1
    return mark
 
def formal_df(ts, df):
    timestamp = df['timestamp'].values
    label = df['label'].values
    data = []
    f = features()
    for i in range(1,len(label)):
        if i % 2000 == 0:
            print i*100.0/len(label),"%"
        tmp =  f.get_features(ts, timestamp[i])
        tmp.append(label[i])
        data.append(tmp)
    return data
from isolation_forest import IForest
def unsupervised(t, df):
    timestamp = df['timestamp'].values
    span = timestamp[1] - timestamp[0]
    w = 3600*3/span
    y = []
    iforest = IForest()
    for i in range(1,len(timestamp)):
        if i % 500 == 0:
            print 'unsupervised:',i*100.0/len(timestamp),"%"
        #构造数据块
        w_now = t.get_series(timestamp[i], w+1)
        yest_stamp = timestamp[i] - 24*3600
        w_yest1 = t.get_series(yest_stamp, w+1)
        yest_stamp2 = timestamp[i] - 24*3600 + 3600*3#后移三个小时
        w_yest2 = t.get_series(yest_stamp2, w)
        last_week = timestamp[i] - 24*3600*7
        w_last_week1 = t.get_series(last_week, w+1)
        last_week2 = timestamp[i] - 24*3600*7 + 3600*3
        w_last_week2 = t.get_series(last_week2, w)
        w_yest1.extend(w_yest2)
        w_last_week1.extend(w_last_week2)
        w_now.extend(w_yest1)
        w_now.extend(w_last_week1)
        r = iforest.predict_score(w_now, w)
        y.append(r)
    return y    
def pack(data, columns):
    data = pd.DataFrame(data, columns= columns)
    print columns
    #data.to_csv('formal.csv')
    return data

def feature_selection(importance):
    features = []
    for i in range(len(importance)):
        features.append([importance[i], i])
    l = sorted(features,key=lambda x:x[0], reverse = False)
    delete_columns = []
    for i in range(len(importance)/3):
        delete_columns.append(l[i][1])
    print delete_columns
    return delete_columns
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
def trainModel(x, y): 
    rf = RandomForestClassifier(n_estimators= 100, class_weight='balanced', max_depth = 12)
    #rf = GradientBoostingClassifier(n_estimators= 100)
    #rf = LogisticRegression(class_weight='balanced', penalty = 'l2')                     
    rf.fit(x,y)
    return rf

from sample import directUse
def trainDirectUse(df,ts, columns):
    data = formal_df(ts, df)
    data = pack(data, columns)
    x,y = directUse(data)
    return trainModel(x, y)

from sample import directDumplicate
def trainDirectDumpli(df, ts, columns, ratio=2):
    data = formal_df(ts, df)
    data = pack(data, columns)
    x,y = directDumplicate(data, ratio)
    return trainModel(x, y)

from sample import sampleNotOnlyT
def trainWeightAnomaly(df, ts, columns, mark, ratio = 2):
    data = formal_df(ts, df)
    data = pack(data, columns)
    x,y = sampleNotOnlyT(data, mark, ratio)
    return trainModel(x, y)

from sample import smote
def trainSmote(df, ts, columns):
    data = formal_df(ts, df)
    data = pack(data, columns)
    x,y = smote(data)
    return trainModel(x, y)

from sample import smoteT
def trainSmoteT(df, ts, columns, mark, ratio = 0.333):
    data = formal_df(ts, df)
    data = pack(data, columns)
    x,y = smoteT(data, mark, ratio)
    return trainModel(x, y)


from sample import dumplicateT
def trainDumpT(df, ts, columns, mark, ratio = 2):
    data = formal_df(ts, df)
    data = pack(data, columns)
    x,y = dumplicateT(data, mark, ratio)
    return trainModel(x, y)
def mixed(df, ts, columns, mark, ratio = 2):
    data = formal_df(ts, df)
    data = pack(data, columns)
    Y = data['label']
    X = data.drop(['label'], axis = 1) 
    x,y = dumplicateT(data, mark, ratio)
    rf_based = trainModel(x, y)
    score1 = rf_based.predict_proba(X)[:,0]
    score1.reshape(1, -1)
    score2 = unsupervised(ts, df)
    score_data = pd.DataFrame()
    score_data['score1'] = pd.Series(score1)
    score_data['score2'] = pd.Series(score2)
    #score_data.to_csv("score.csv")
    rf_top = trainModel(score_data, Y)
    return rf_based, rf_top
    