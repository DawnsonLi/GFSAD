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
def show_train(y_train, df):
    x = pd.read_csv('formal.csv')
    x= x.drop(['label'], axis = 1).values
    x = np.delete(x, 1, axis=1)
    predict_train = rf.predict(x)
    evalute_delay(y_train, predict_train)
    ans = truth_predict(y_train, predict_train)
    dic = {'y':ans}
    predict = pd.DataFrame(dic)
    predict['timestamp'] = df['timestamp']
    predict['value'] = df['value']
    plot_ans(predict, ratio = 1)
    
def show_test(test, predict_test):
    y_test = test['label']
    evalute_delay(y_test, predict_test)
    ans = truth_predict(y_test, predict_test)
    dic = {'y':ans}
    predict = pd.DataFrame(dic)
    predict['timestamp'] = test['timestamp']
    predict['value'] = test['value']
    plot_ans(predict, ratio = 1)
    

from TSlist import TSlist
df = pd.read_csv('data/series/7_train.csv')
span = df['timestamp'][1] - df['timestamp'][0]
ts = TSlist(df)
df = df[3600*24/span+5:]
print 'start to fill missed value'
ts.fill_missed_median()

print 'filling over'
y_train = df['label'].values
mark = extract_Tsamples(y_train)
columns = ['value','mean_now', 'std_now', 'minus_mean_now_past', 'minus_mean_now_yes','minus_std_now_past', 'minus_std_now_yes', 'r_mean_now_past', 'r_mean_now_yes','1', '2', '3','4','5','6','7','8','ewma','pre_minus','pre_rate','yes_minus','yes_rate','label']
print 'start to train'
rf  = trainDumpT(df, ts, columns, mark, ratio = 3)

#show_train(y_train, df)
print 'training done'
test = pd.read_csv('data/series/7_test.csv')
ts.append_df(test)
ts.fill_missed_median()
predict_test = benchmark(ts, test, rf)
#show_test(test, predict_test)
print evalute_delay(test['label'], predict_test)