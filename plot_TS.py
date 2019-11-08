#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 15:49:36 2019

@author: dine
"""

import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib import style



def mapping(x):
    if x in [2,3]:
        return 2
    else:
        return x
    
def mapping2(x):
    if x==0:
        return '1st'
    elif x==1:
        return '2nd'
    elif x==2:
        return '3rd'
    else:
        return '4th'

def plotstuff():
    with open('results_HB_new2.pkl', 'rb') as handle:
        res = pickle.load(handle)
    #%matplotlib qt
    
#    tab = pd.DataFrame()
#    for arm in res.arms:
#        tab = pd.concat([tab,arm.hb.evals])
    tab = res
    etas = [x[-1] for x in tab.conf]

    tab['s'] = 0
    tab.reset_index(drop=True, inplace=True)
    tab['s'] = (tab.index)%5
    tab.s = tab.s.apply(lambda x: mapping(x))
    tab.s = tab.s.apply(lambda x: mapping2(x))
    tab['eta'] = etas
    tab = tab[tab.L>0.6]

    print(tab)

    style.use('fivethirtyeight')

    fig = plt.figure()
    ax1 = plt.subplot2grid((10,10), (0,0), rowspan=2, colspan=8)
    for elem in tab.s.unique():    
        sns.kdeplot(tab.eta[tab.s == elem])
    ax1.get_legend().remove()
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)

    ax2 = plt.subplot2grid((10,10), (2,0), rowspan=8, colspan=8, sharex=ax1)
    ax2.legend('Brackets')
    #plt.title('stock')
    #plt.ylabel('H-L')
    a = sns.scatterplot(tab.eta, tab.L, hue=tab.s)
    
    ax3 = plt.subplot2grid((10,10), (2,8), rowspan=8, colspan=2, sharey=ax2)
    for elem in tab.s.unique():    
        sns.kdeplot(tab.L[tab.s == elem], vertical=True)
    #plt.ylabel('Price')
    #plt.ylabel('MAvgs')
    ax3.get_legend().remove()
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)

    plt.show()


plotstuff()


