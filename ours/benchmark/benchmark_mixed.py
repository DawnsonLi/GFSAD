# -*- coding: utf-8 -*-
from train import *
from visio import truth_predict
from evalute import evalute_delay
from visio import plot_ans
from isolation_forest import IForest
def benchmark(ts, df, rf, rf_top):
    timestamp = df['timestamp'].values
    f = features()
    y = []
    w = 3600*3/ts.span
    iforest = IForest()
    for i in range(1,len(timestamp)):
        if i % 5000 == 0:
            print i*100.0/len(timestamp),"%"
        tmp = rf.predict_proba(np.array(f.get_features(ts, timestamp[i])).reshape(1, -1))
        score1 =  tmp[0][0]
        #构造数据块
        w_now = ts.get_series(timestamp[i], w+1)
        yest_stamp = timestamp[i] - 24*3600
        w_yest1 = ts.get_series(yest_stamp, w+1)
        yest_stamp2 = timestamp[i] - 24*3600 + 3600*3#后移三个小时
        w_yest2 = ts.get_series(yest_stamp2, w)
        last_week = timestamp[i] - 24*3600*7
        w_last_week1 = ts.get_series(last_week, w+1)
        last_week2 = timestamp[i] - 24*3600*7 + 3600*3
        w_last_week2 = ts.get_series(last_week2, w)
        w_yest1.extend(w_yest2)
        w_last_week1.extend(w_last_week2)
        w_now.extend(w_yest1)
        w_now.extend(w_last_week1)
        score2 = iforest.predict_score(w_now, w)
        x = np.array([score1, score2]).reshape(1, -1)
        y.append( rf_top.predict(x))
    return y
from TSlist import TSlist
file_name = 'data/series/'
numbers = ['26']
#numbers = ['8','10','19','27']
log = []
columns = ['value','mean_now', 'std_now', 'minus_mean_now_past', 'minus_mean_now_yes','minus_std_now_past', 'minus_std_now_yes', 'r_mean_now_past', 'r_mean_now_yes','1', '2', '3','4','5','6','7','8','ewma','pre_minus','pre_rate','yes_minus','yes_rate','label']
for num in numbers:
    print num
    df = pd.read_csv(file_name+num+'_train.csv')
    span = df['timestamp'][1] - df['timestamp'][0]
    ts = TSlist(df)
    df = df[3600*24*7/span+15:]
    print 'start to fill missed value'
    ts.fill_missed_median()
    print 'filling over'
    y_train = df['label'].values
    mark = extract_Tsamples(y_train)
    print 'start to train'
    #----------
    rf_based, rf_top = mixed(df, ts, columns, mark, ratio = 3)
    print 'training done'
    test = pd.read_csv(file_name+num+'_test.csv')
    ts.append_df(test)
    ts.fill_missed_median()
    predict_test = benchmark(ts, test, rf_based, rf_top)
    
    precison, recall, f1  = evalute_delay(test['label'], predict_test)
    log.append([num, precison, recall, f1])
    log_tmp = pd.DataFrame(log, columns=['kpi id','precison','recall','fscore'])
    log_tmp.to_csv(num+'mixed.csv')
    

