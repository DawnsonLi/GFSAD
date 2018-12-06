# -*- coding: utf-8 -*-
import pandas as pd
from imblearn.over_sampling import SMOTE
'''
data: dataframe类型
mark: list类型
'''
def smote(data, ratio = 0.333):
    smo = SMOTE(random_state=42,sampling_strategy=ratio)
    y = data['label']
    x = data.drop(['label'], axis = 1) 
    print  'before smote:',len(x)
    X_smo, y_smo = smo.fit_sample(x, y)
    print  'after smote:',len(X_smo)
    return X_smo, y_smo

def smoteT(data, mark, ratio = 0.333):
    smo = SMOTE(random_state=42,sampling_strategy=ratio)
    data['mark'] = pd.Series(mark)
    data = data[ data['mark'] != 2 ]
    y = data['label']
    x = data.drop(['label','mark'], axis = 1) 
    print  'before smote:',len(x)
    X_smo, y_smo = smo.fit_sample(x, y)
    print  'after smote:',len(X_smo)
    return X_smo, y_smo

def directUse(data):
    y = data['label']
    x = data.drop(['label'], axis = 1) 
    return x, y
'''
dumplicate all the data
'''
def directDumplicate(data,ratio = 2):
    anomalyT = data[ data['label'] == 1]
    normal = data[ data['label'] == 0 ]
    print len(normal), "size of normal sample"
    times = len(normal)/(ratio*len(anomalyT))
    for i in range(int(times)):
        normal = normal.append(anomalyT)
    print len(normal), "size of all after sampled"
    y = normal['label']
    x = normal.drop(['label'], axis = 1) 
    return x, y
'''
use all anomaly data for sample, but data inside window has bigger weight
'''
def sampleNotOnlyT(data, mark, ratio = 2):
    data['mark'] = pd.Series(mark)
    anomalyT = data[ data['mark'] == 1]
    normal = data[ data['mark'] == 0 ]
    anomalyOutT = data[ data['mark'] == 2]
    weights = [80 for i in range(len(anomalyT))].extend([1 for j in range(len(anomalyOutT))])
    anomalyT = anomalyT.append(anomalyOutT)
    print len(normal), "size of normal sample"
    try:
        #replace：是否为有放回抽样，取replace=True时为有放回抽样。
        anomalysample = anomalyT.sample(weights = weights, n = len(normal)/ratio,  replace = True)
        normal = normal.append(anomalysample)
    except:
        print 'sample meets a trouble'
    print len(normal), "size of all after sampled"
    y = normal['label']
    x = normal.drop(['mark','label'], axis = 1) 
    return x, y

'''
sample the data which falls in time window, and ignore the data outside window
'''
def sampleT(data, mark, ratio = 2):
    data['mark'] = pd.Series(mark)
    anomalyT = data[ data['mark'] == 1]
    normal = data[ data['mark'] == 0 ]
    print len(normal), "size of normal sample"
    try:
        anomalysample = anomalyT.sample( n = len(normal)/ratio - len(anomalyT),  replace = True)
        normal = normal.append(anomalysample)
    except:
        print 'sample meets a trouble'
    normal = normal.append(anomalyT)
    print len(normal), "size of all after sampled"
    y = normal['label']
    x = normal.drop(['mark','label'], axis = 1) 
    return x, y
'''
dumplicate the data which falls in time window, and ignore the data outside window
'''
def dumplicateT(data, mark, ratio = 2):
    data['mark'] = pd.Series(mark)
    anomalyT = data[ data['mark'] == 1]#编码见baseutil.py
    normal = data[ data['mark'] == 0 ]
    print len(normal), "size of normal sample"
    times = len(normal)/(ratio*len(anomalyT))
    for i in range(int(times)):
        normal = normal.append(anomalyT)
    print len(normal), "size of all after sampled"
    y = normal['label']
    x = normal.drop(['mark','label'], axis = 1) 
    return x, y
