# -*- coding: utf-8 -*-
from train import *
from visio import truth_predict
from evalute import evalute_delay
from visio import plot_ans

def benchmark(ts, df, rf ):
    timestamp = df['timestamp'].values
    f = features()
    y = []
    for i in range(1,len(timestamp)):
        if i % 5000 == 0:
            print i*100.0/len(timestamp),"%"
        y.append( rf.predict(np.array(f.get_features(ts, timestamp[i])).reshape(1, -1))[0])
    return y
from TSlist import TSlist
file_name = 'data/series/'
numbers = ['14']
#numbers = ['2','6','7','10','27','26','14']
log = []
columns = ['value','mean_now', 'std_now', 'minus_mean_now_past', 'minus_mean_now_yes','minus_std_now_past', 'minus_std_now_yes', 'r_mean_now_past', 'r_mean_now_yes','1', '2', '3','4','5','6','7','8','ewma','pre_minus','pre_rate','yes_minus','yes_rate','label']
for num in numbers:
    print num
    df = pd.read_csv(file_name+num+'_train.csv')
    span = df['timestamp'][1] - df['timestamp'][0]
    ts = TSlist(df)
    df = df[3600*24/span+15:]
    print 'start to fill missed value'
    ts.fill_missed_median()
    print 'filling over'
    print 'start to train'
    y_train = df['label'].values
    mark = extract_Tsamples(y_train)
    rf = trainDirectUse(df,ts, columns)
    print 'training done'
    test = pd.read_csv(file_name+num+'_test.csv')
    ts.append_df(test)
    ts.fill_missed_median()
    predict_test = benchmark(ts, test, rf)
    
    precison, recall, f1  = evalute_delay(test['label'], predict_test)
    log.append([num, precison, recall, f1])
    log_tmp = pd.DataFrame(log, columns=['kpi id','precison','recall','fscore'])
    log_tmp.to_csv(num+'direct.csv')
    
