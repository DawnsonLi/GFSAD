# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from polynomial_interpolation import *
class features(object):
    def _statis(self, s):
        if len(s) > 0:
            return np.mean(s), np.std(s)
        return 0, 0

    def _zscore(self, s, value):
        avg = np.mean(s)
        s.append(value)
        std = np.std(s)
        if std <= 0.00000001:
            return 0
        return (value - avg)/std
    def _change_rate(self, v_pre, v_now):
        small = 0.0000001
        return (v_now - v_pre)/(v_pre + small)
    '''
        与窗口w相关的特征
    '''
    def _window_features(self, ts, timestamp, w):
        s_now, s_past, s_yesterday = ts.get_three_series(timestamp, w)
        mean_now, std_now = self._statis(s_now)
        mean_past, std_past = self._statis(s_past)
        mean_yes, std_yes = self._statis(s_yesterday)
        #extract minus feature
        minus_mean_now_past = mean_now - mean_past
        minus_mean_now_yes = mean_now - mean_yes
        minus_std_now_past = std_now - std_past
        minus_std_now_yes = std_now - std_yes
        #extract rate feature
        rate_mean_now_past = self._change_rate(mean_past, mean_now)
        rate_mean_now_yes = self._change_rate(mean_yes, mean_now)
        #rate_std_now_past = self._change_rate(std_past, std_now)
        #rate_std_now_yes = self._change_rate(std_yes, std_now)
        #zscore
        #zscore_yes = self._zscore(s_yesterday, value)
        win_feature = [mean_now, std_now]
        win_feature.extend([minus_mean_now_past, minus_mean_now_yes, minus_std_now_past , minus_std_now_yes])
        win_feature.extend([rate_mean_now_past, rate_mean_now_yes])
        #win_feature.append(zscore_yes)
        return win_feature
    
    
    def _window_all_features(self,ts, timestamp, w):
        s_now,s_past, s_yesterday, s_lastweek = ts.get_four_series(timestamp, w=3)
        mean_now, std_now = self._statis(s_now)
        mean_past, std_past = self._statis(s_past)
        mean_yes, std_yes = self._statis(s_yesterday)
        mean_lw, std_lw = self._statis(s_lastweek)
        #extract minus feature
        minus_mean_now_past = mean_now - mean_past
        minus_mean_now_yes = mean_now - mean_yes
        minus_mean_now_lw = mean_now - mean_lw
        minus_std_now_past = std_now - std_past
        minus_std_now_yes = std_now - std_yes
        minus_std_now_lw = std_now - std_lw
        #extract rate feature
        rate_mean_now_past = self._change_rate(mean_past, mean_now)
        rate_mean_now_yes = self._change_rate(mean_yes, mean_now)
        rate_mean_now_lw = self._change_rate(mean_lw, mean_now)
        #rate_std_now_past = self._change_rate(std_past, std_now)
        #rate_std_now_yes = self._change_rate(std_yes, std_now)
        #zscore
        #zscore_yes = self._zscore(s_yesterday, value)
        win_feature = [mean_now, std_now]
        win_feature.extend([minus_mean_now_past, minus_mean_now_yes, minus_mean_now_lw, minus_std_now_past , minus_std_now_yes,minus_std_now_lw])
        win_feature.extend([rate_mean_now_past, rate_mean_now_yes, rate_mean_now_lw])
        #win_feature.append(zscore_yes)
        return win_feature
    
    '''
        获取拟合特征
    '''
    def _ewma_feature(self, ts, timestamp, w):
        s_now, s_yes = ts.get_two_series(timestamp, w)
        s_yes.extend(s_now)
        s = self._ewma(s_yes)
        return s_now[-1] - s[-1]
    
    def _ewma(self, X, alpha=0.3):
        s = [X[0]]
        for i in range(1, len(X)):
            temp = alpha * X[i] + (1 - alpha) * s[-1]
            s.append(temp)
        return s
     
    def get_features(self, ts, timestamp):
        tmp = []
        value = ts.get_value(timestamp)
        tmp.append(value)
        #window feature
        tmp.extend( self._window_features(ts, timestamp, w=3))
        tmp.extend( self._window_features(ts, timestamp, w=5))
        #print window_features(ts, timestamp, w=5)
        #print window_features(ts, timestamp, w=7)
        #fitting feature
        tmp.append( self._ewma_feature(ts, timestamp, w = 36))
        #previous point contrast feature
        pre_value = ts.get_previous_value(timestamp)
        tmp.append( value - pre_value)
        tmp.append( self._change_rate(pre_value, value))
        #last day
        yest_day_value = ts.get_yest_value(timestamp)
        tmp.append( value - yest_day_value)
        tmp.append( self._change_rate(yest_day_value, value))
        return tmp
        