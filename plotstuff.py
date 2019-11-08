import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib import style



def mapping(x):
    if x>3:
        return 3
    elif x in [2,3]:
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
    with open('results.pkl', 'rb') as handle:
        res = pickle.load(handle)
    %matplotlib qt

    etas = [x[-1] for x in res.evals.conf]

    tab = res.evals
    tab['s'] = 0
    tab.reset_index(drop=True, inplace=True)
    tab['s'] = (tab.index)%8
    tab.s = tab.s.apply(lambda x: mapping(x))
    tab.s = tab.s.apply(lambda x: mapping2(x))
    tab['eta'] = etas
    tab = tab[tab.L>0.6]



    style.use('fivethirtyeight')

    fig = plt.figure()
    ax1 = plt.subplot2grid((10,10), (0,0), rowspan=2, colspan=8)
    for elem in tab.s.unique():    
        sns.kdeplot(tab.eta[tab.s == elem])    
    ax2 = plt.subplot2grid((10,10), (2,0), rowspan=8, colspan=8, sharex=ax1)
    #plt.title('stock')
    #plt.ylabel('H-L')
    sns.scatterplot(tab.eta, tab.L, hue=tab.s)
    ax3 = plt.subplot2grid((10,10), (2,8), rowspan=8, colspan=2, sharey=ax2)
    for elem in tab.s.unique():    
        sns.kdeplot(tab.L[tab.s == elem], vertical=True)
    #plt.ylabel('Price')
    #plt.ylabel('MAvgs')
    plt.show()


plotstuff()

