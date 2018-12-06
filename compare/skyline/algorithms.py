# -*- coding: utf-8 -*-
import pandas
import numpy as np
import scipy
import scipy.stats as stats
import statsmodels.api as sm
from time import time
from algorithm_exceptions import *
from dask.datasets import timeseries
"""
This is no man's land. Do anything you want in here,
as long as you return a boolean that determines whether the input
timeseries is anomalous or not.
To add an algorithm, define it here, and add its name to settings.ALGORITHMS.
"""

def tail_avg(timeseries):
    """
    This is a utility function used to calculate the average of the last three
    datapoints in the series as a measure, instead of just the last datapoint.
    It reduces noise, but it also reduces sensitivity and increases the delay
    to detection.
    """
    try:
        t = (timeseries[-1] + timeseries[-2] + timeseries[-3]) / 3
        return t
    except IndexError:
        return timeseries[-1]


def median_absolute_deviation(timeseries):
    """
    A timeseries is anomalous if the deviation of its latest datapoint with
    respect to the median is X times larger than the median of deviations.
    """
    median = np.median(timeseries)
    demedianed =  []
    for t in timeseries:
        demedianed.append( abs(t - median))
    median_deviation = np.median(demedianed)
    # The test statistic is infinite when the median is zero,
    # so it becomes super sensitive. We play it safe and skip when this happens.
    if median_deviation == 0:
        return False
    test_statistic = demedianed[-1] / median_deviation

    # Completely arbitary...triggers if the median deviation is
    # 6 times bigger than the median
    if test_statistic > 6:
        return True


def grubbs(timeseries):
    """
    A timeseries is anomalous if the Z score is greater than the Grubb's score.
    """

    series = scipy.array(timeseries)
    stdDev = scipy.std(series)
    mean = np.mean(series)
    tail_average = tail_avg(timeseries)
    z_score = (tail_average - mean) / stdDev
    len_series = len(series)
    threshold = stats.t.isf(.05 / (2 * len_series), len_series - 2)
    threshold_squared = threshold * threshold
    grubbs_score = ((len_series - 1) / np.sqrt(len_series)) * np.sqrt(threshold_squared / (len_series - 2 + threshold_squared))

    return z_score > grubbs_score


def first_hour_average(timeseries):
    """
    Calcuate the simple average over one hour, FULL_DURATION seconds ago.
    A timeseries is anomalous if the average of the last three datapoints
    are outside of three standard deviations of this value.
    Attention: to control timeseries in an hour
    """
    series = pandas.Series(timeseries)
    mean = series.mean()
    stdDev = series.std()
    t = tail_avg(timeseries)

    return abs(t - mean) > 3 * stdDev


def stddev_from_average(timeseries):
    """
    A timeseries is anomalous if the absolute value of the average of the latest
    three datapoint minus the moving average is greater than three standard
    deviations of the average. This does not exponentially weight the MA and so
    is better for detecting anomalies with respect to the entire series.
    """
    series = pandas.Series(timeseries)
    mean = series.mean()
    stdDev = series.std()
    t = tail_avg(timeseries)

    return abs(t - mean) > 3 * stdDev

def ewma( X, alpha=0.3):
        s = [X[0]]
        for i in range(1, len(X)):
            temp = alpha * X[i] + (1 - alpha) * s[-1]
            s.append(temp)
        return s
    
def stddev_from_moving_average(timeseries):
    """
    A timeseries is anomalous if the absolute value of the average of the latest
    three datapoint minus the moving average is greater than three standard
    deviations of the moving average. This is better for finding anomalies with
    respect to the short term trends.
    """
    s = ewma(timeseries)
    expAverage = np.mean(s)
    stdDev = np.std(s)

    return abs(timeseries[-1] - expAverage) > 3 * stdDev


def mean_subtraction_cumulation(timeseries):
    """
    A timeseries is anomalous if the value of the next datapoint in the
    series is farther than three standard deviations out in cumulative terms
    after subtracting the mean from each data point.
    """

    series = []
    mean = np.mean(timeseries)
    for t in timeseries:
        series.append( t - mean)
    stdDev = np.std(series)
    #expAverage = pandas.stats.moments.ewma(series, com=15)

    return abs(series[-1]) > 3 * stdDev


def least_squares(timeseries):
    """
    A timeseries is anomalous if the average of the last three datapoints
    on a projected least squares model is greater than three sigma.
    """

    x = np.array( [i for i in range(len(timeseries))])
    y = np.array(timeseries)
    A = np.vstack([x, np.ones(len(x))]).T
    #results = np.linalg.lstsq(A, y)
    #residual = results[1]
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    errors = []
    for i in range(len(y)):
        projected = m * x[i] + c
        error = y[i] - projected
        errors.append(error)

    if len(errors) < 3:
        return False

    std_dev = scipy.std(errors)
    t = (errors[-1] + errors[-2] + errors[-3]) / 3

    return abs(t) > std_dev * 3 and round(std_dev) != 0 and round(t) != 0


def histogram_bins(timeseries):
    """
    A timeseries is anomalous if the average of the last three datapoints falls
    into a histogram bin with less than 20 other datapoints (you'll need to tweak
    that number depending on your data)
    Returns: the size of the bin which contains the tail_avg. Smaller bin size
    means more anomalous.
    """

    series = scipy.array(timeseries)
    t = tail_avg(timeseries)
    h = np.histogram(series, bins=15)
    bins = h[1]
    for index, bin_size in enumerate(h[0]):
        if bin_size <=5:
            # Is it in the first bin?
            if index == 0:
                if t <= bins[0]:
                    return True
            # Is it in the current bin?
            elif t >= bins[index] and t < bins[index + 1]:
                    return True

    return False


def ks_test(reference, probe):
    """
    A timeseries is anomalous if 2 sample Kolmogorov-Smirnov test indicates
    that data distribution for last 10 minutes is different from last hour.
    It produces false positives on non-stationary series so Augmented
    Dickey-Fuller test applied to check for stationarity.
    """

    #hour_ago = time() - 3600
    #ten_minutes_ago = time() - 600
    #reference = scipy.array([x[1] for x in timeseries if x[0] >= hour_ago and x[0] < ten_minutes_ago])
    #probe = scipy.array([x[1] for x in timeseries if x[0] >= ten_minutes_ago])

    if len(reference)< 20 or len(probe) < 20:
        return False

    ks_d, ks_p_value = stats.ks_2samp(reference, probe)

    if ks_p_value < 0.05 and ks_d > 0.5:
        adf = sm.tsa.stattools.adfuller(reference, 10)
        if adf[1] < 0.05:
            return True

    return False

import logging
def benchmark(ts, df, threshold=6):
        timestamp = df['timestamp'].values
        span = ts.span
        y = []
        for i in range(len(timestamp)):
            if i % 5000 == 0:
                print i*100.0/len(timestamp),"%"
            #y.append( self.statistic(ts.get_series(timestamp[i], span)))
            one_day = ts.get_series(timestamp[i], 3600*24/span)
            three_hour = ts.get_series(timestamp[i], 3600*3/span)
            ks = ts.get_series(timestamp[i], 3600*2/span)
            counter = 0
            if median_absolute_deviation(one_day):
                counter += 1
            if grubbs(one_day):
                counter += 1
            if first_hour_average(three_hour):
                counter += 1
            if stddev_from_average(one_day):
                counter += 1
            if stddev_from_moving_average(one_day):
                counter += 1
            if mean_subtraction_cumulation(one_day):
                counter += 1
            if least_squares(one_day):  
                counter += 1
            if histogram_bins(one_day):
                counter += 1
            if ks_test(three_hour, ks):
                counter += 1  
            if counter >= threshold:
                y.append(1)#anomaly
            else:
                y.append(0)    
        return y
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
    if tp <=0:
        f1 = 0
        precison = 0
        recall = 0
    else:
        precison = tp*1.0/(tp+fp)
        recall = tp*1.0/(tp+fn)
        f1 = 2*precison*recall/(precison+recall)
    print 'precison:',precison
    print 'recall:',recall
    print 'f1 score:',f1     
    return  precison, recall, f1
import pandas as pd
from TSlist import TSlist
 
file_name = 'E:/javacode/AIOPS/data/series/'
numbers = ['14']
#numbers = ['2','6','7','10','19','27']
log = []
for num in numbers:
    print 'kpi:',num
    train = pd.read_csv(file_name+num+'_train.csv')
    t = TSlist(train)
    test = pd.read_csv(file_name+num+'_test.csv')
    t.append_df(test)
    t.fill_missed_median()
    y = benchmark(t, test)
    try:
        precison, recall, f1  = evalute_delay(test['label'], y)
        log.append([num, precison, recall, f1])
    except:
        print 'trouble'
    log_tmp = pd.DataFrame(log, columns=['kpi id','precison','recall','fscore'])
    log_tmp.to_csv(num+'skyline.csv')
    del train
    del test
    del t