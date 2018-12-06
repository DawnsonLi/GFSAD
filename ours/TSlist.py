# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from numpy import median
class TSlist:
    start_stamp = 0 #该序列起始时间
    ts = []
    span = 5 #采样的时间差
    second_week_stamp = 0 #第二周的同时刻的起始时间
    second_day_stamp = 0 #第二天的同时刻的起始时间
    def __init__(self, df):
        values = df['value'].values
        timestamp = df['timestamp'].values
        tmp = {}
        for i in range(len(values)):
            tmp[timestamp[i]] = values[i]
        self.start_stamp = timestamp[0]
        self.span = timestamp[1] - timestamp[0]
        self.second_day_stamp = self.start_stamp + 3600*24
        self.second_week_stamp = self.start_stamp + 3600*24*7
        for t in range(self.start_stamp, timestamp[len(timestamp)-1]+self.span, self.span):
            if t in tmp:
                self.ts.append(tmp[t])
            else:
                self.ts.append("missed")
        del tmp
    def _last_stamp(self):
        return self.start_stamp + self.span*(len(self.ts)-1)
    def append_df(self, df):
        values = df['value'].values
        timestamp = df['timestamp'].values
        tmp = {}
        for i in range(len(values)):
            tmp[timestamp[i]] = values[i]
        start_stamp = timestamp[0]
        last_stamp = self._last_stamp()
        for t in range(last_stamp, start_stamp, self.span):
            self.ts.append("missed")
        for t in range(start_stamp, timestamp[len(timestamp)-1]+self.span, self.span):
            if t in tmp:
                self.ts.append(tmp[t])
            else:
                self.ts.append("missed")
        del tmp
    def get_ts(self):
        return self.ts

    def has_missed(self):
        if "missed" in self.ts:
            return True
        return False
    
    def get_value(self, timestamp):
        index = (timestamp - self.start_stamp)/self.span
        if index < 0:
            return 
        return self.ts[index]
    def get_previous_value(self, timestamp):
        previous_stamp = timestamp - self.span
        return self.get_value(previous_stamp)
     
    def get_yest_value(self, timestamp):
        previous_stamp = timestamp - 3600*7
        return self.get_value(previous_stamp)
    
    def get_lastweek_value(self, timestamp):
        timestamp_lastweek = timestamp - 3600*24*7
        return self.get_value(timestamp_lastweek)
    def get_index(self, timestamp):
        index = (timestamp - self.start_stamp)/self.span
        if index < 0:
            return -1
        return index
    def get_series(self, timestamp, w=5):
        index = (timestamp - self.start_stamp)/self.span
        if index < 0:
            return []
        if index < w:
            return self.ts[:index+1]
        return self.ts[index-w+1:index+1]
     
    def get_four_series(self, timestamp, w=5):
        timestamp_yesterday = timestamp - 3600*24
        timestamp_lastweek = timestamp - 3600*24*7
        timestamp_past = timestamp - w*self.span    
        s_now = self.get_series(timestamp, w)
        s_past = self.get_series(timestamp_past, w)
        s_yesterday = self.get_series(timestamp_yesterday, w)
        s_lastweek = self.get_series(timestamp_lastweek, w)
        return s_now,s_past, s_yesterday, s_lastweek
       
    def get_three_series(self, timestamp, w=5):
        timestamp_past = timestamp - w*self.span
        timestamp_yesterday = timestamp - 3600*24
        s_now = self.get_series(timestamp, w)
        s_past = self.get_series(timestamp_past, w)
        s_yesterday = self.get_series(timestamp_yesterday, w)
        return s_now, s_past, s_yesterday
    
    def get_two_series(self, timestamp, w=5):
        timestamp_yesterday = timestamp - 3600*24
        s_now = self.get_series(timestamp, w)
        s_yesterday = self.get_series(timestamp_yesterday, w)
        return s_now, s_yesterday
    def fill_value(self, index, count):
        l = []
        step = 3600*24/self.span
        i = index - step
        while len(l) < count:
            if i >= 0: 
                if self.ts[i] != "missed":
                    l.append(self.ts[i])
                i -= step
            if i < 0:
                i = index + step
                break
        while len(l) < count:
            if i <len(self.ts): 
                if self.ts[i] != "missed":
                    l.append(self.ts[i])
                i += step
            if i >= len(self.ts):
                break
        #print l
        return median(l)            
    '''
       使用count个点同周期点的中位数进行填充
    '''  
    def fill_missed_median(self, count=5):
        counter = 0
        for i in range(len(self.ts)):
            if self.ts[i] == "missed":
                self.ts[i] = self.fill_value(i, count) 
                counter += 1
        return counter
    def fill_missed_all_avg(self):
        avg = np.mean([v for v in self.ts if v != 'missed'])
        for i in range(len(self.ts)):
            if self.ts[i] == "missed":
                self.ts[i] = avg 
        
