# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 16:58:19 2019

@author: dinello
"""
from Hyperband import HyperBand
import numpy as np
from scipy.stats import norm, invgamma

class Arm(object):
    def __init__(self,hb, mu, n, nu, sigma2):
        self.hb = hb
        self.improvement = 100
        self.mu0 = mu
        self.n0 = n
        self.nu0 = nu
        self.sigma2_0 = sigma2
        self.n = 0
        self.mu1 = mu
        self.n1 = n
        self.nu1 = nu
        self.sigma2_1 = sigma2

    def compute_posterior(self):
        losses = self.hb.evals['L']
        meanloss = np.mean(losses)
        num = len(losses)
        self.n1 = self.n0 + num
        self.mu1 = (self.n0*self.mu0 + meanloss*num)/self.n1
        self.nu1 = self.nu0 + num
        S = np.sum((losses-meanloss)**2)
        self.sigma2_1 = (self.nu0*self.sigma2_0 + S + self.n0*num/(self.n0 + num)*(self.mu0-meanloss)**2)/self.nu1
        return
    
    def compute_probability(self, currmax):
        std = invgamma(a=self.nu1/2).rvs()**0.5
        mean = norm(self.mu1, std).rvs()
        self.improvement = self.expected_improvement(currmax,mean,std)
        print("EI for arm: ", self.improvement)
        return
    
    def expected_improvement(self, currmax, mn, st):
        x = (mn - currmax)/st
        pdf = norm.pdf(x)
        cdf = norm.cdf(x)
        return st*pdf + abs(mn-currmax)*cdf
