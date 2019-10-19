# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 16:58:19 2019

@author: dinello
"""
from Hyperband import HyperBand
import numpy as np
from scipy.stats import t
from math import ceil

class arm(object):
    def __init__(self,hb):
        self.hb = hb
        self.best_mean = 1
        
    def compute_best_mean(self):
        mysort = self.hb.evals.sort_values('L')
        k = ceil(mysort.shape[0]/5)
        self.best_mean = mysort.iloc[-k:].L.mean()

        