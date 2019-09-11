# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 16:58:19 2019

@author: dinello
"""
from Hyperband import HyperBand
import numpy as np

class Arm(object, mu, n, nu, sigma2):
    def __init__(self,hb):
        self.hb = hb
        self.probability = 100
        self.mu0 = mu
        self.n0 = n
        self.nu0 = nu
        self.sigma2_0 = sigma2
        self.n = 0
        self.mu1 = mu
        self.n1 =n
        self.nu1 = nu
        self.sigma2_1 = sigma2

    def compute_posterior(self):
        losses = self.hb.evals['L']
        meanloss = np.mean(losses)
        num = len(losses)
        self.n1 = n0 + n
        self.mu1 = (self.n0*self.mu0 + meanloss*num)/self.n1
        self.nu1 = self.nu0 + num
        S = np.sum((losses-meanloss)**2)
        self.sigma_1 = (self.nu0*self.sigma_0 + S + self.n0*num/(self.n0 + num)*(self.mu0-meanloss)**2)/self.nu1
        
    def compute_probability(self,currmax)