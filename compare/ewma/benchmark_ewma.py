# -*- coding: utf-8 -*-
from TSlist import TSlist
import pandas as pd
def to_str(l):
    s = ""
    for i in range(len(l)-1):
        s += str(l[i])
        s += ","
    s += str(l[len(l)-1])
    return s
def evalute_delay(real, predict, delay = 7):
    counter = 0#记录异常片段的长度
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    find = False
    for i in range(len(predict)):
        if real[i] == 0:#正常数据
            if find:
                tp += counter
                counter = 0
                find = False
            else:
                fn += counter
                counter = 0              
            if predict[i] == 0:
                tn += 1
            else:
                fp += 1
        else:
            if predict[i] != 0 and counter <= delay:
                find = True
            counter += 1
            #print counter
    if counter > 0 and find:#处理尾部是异常的情况
        tp += counter  
    if counter >0 and find == False:
        fn += counter
    
    print 'tp:',tp  
    print 'tn:',tn 
    print 'fp:',fp 
    print 'fn:',fn  
    precison = tp*1.0/(tp+fp)
    recall = tp*1.0/(tp+fn)
    if tp ==0:
        f1 = 0
    else:
        f1 = 2*precison*recall/(precison+recall)
    print 'precison:',precison
    print 'recall:',recall
    print 'f1 score:',f1
    return precison, recall, f1 
    
from ewma import Detect
file_name = 'E:/javacode/AIOPS/data/series/'
numbers = ['14']
#numbers = ['2','6','7','10','19','27']
log = []
for num in numbers:
    print num
    train = pd.read_csv(file_name+num+'_train.csv')
    t = TSlist(train)
    test = pd.read_csv(file_name+num+'_test.csv')
    t.append_df(test)
    t.fill_missed_median()
    timestamp = test['timestamp'].values
    span = timestamp[1] - timestamp[0]
    y = []
    w = 3600*3/span
    for i in range(1,len(timestamp)):
        if i % 5000 == 0:
            print i*100.0/len(timestamp),"%"
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
        dataA = to_str(w_now)
        dataB = to_str(w_yest1)
        dataC = to_str(w_last_week1)
        d = {"viewId":"2012","viewName":"登陆功能","attrId":"19201","attrName":"ptlogin登陆请求总量", "window":w,"time":"2018-10-17 17:28:00",}
        d["dataC"] = dataC
        d["dataB"] = dataB
        d["dataA"] = dataA
        
        detector =  Detect()
        TSD_OP_SUCCESS, ret_data = detector.value_predict(d)
        if ret_data['ret'] > 0.5: #正常
            y.append(0)
        else:
            y.append(1)
        
    precison, recall, f1= evalute_delay(test['label'], y, delay = 7)
    log.append([num, precison, recall, f1])
    log = pd.DataFrame(log, columns=['kpi id','precison','recall','fscore'])
    log.to_csv(num+'ewma.csv')
    del train
    del test 
    del t


'''
@20
tp: 160
tn: 3714
fp: 6878
fn: 27
precison: 0.0227337311736
recall: 0.855614973262
f1 score: 0.0442906574394

@7
tp: 1610
tn: 108759
fp: 25
fn: 975
precison: 0.984709480122
recall: 0.622823984526
f1 score: 0.763033175355
'''