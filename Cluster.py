# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 11:30:06 2017

@author: Hao Guo
"""

import datetime as dt
import matplotlib.pyplot as plt 
import numpy as np 
import numpy.linalg as la 
import pandas as pd
import scipy.cluster.hierarchy as sch
import seaborn as sns
import pandas_datareader.data as web
from scipy.stats import mode 
from nearest_correlation import nearcorr




tickerdf = pd.read_csv('SandP500.csv')    
tickers = list(tickerdf['Symbol'].values) 
verbose_flag = False                      
start_date = dt.datetime(2011, 1, 1)      
ticker_df_list = []                                                 

for ticker in tickers: 
    try:
        r = web.DataReader(ticker, "yahoo", start=start_date)   
        r['Ticker'] = ticker 
        ticker_df_list.append(r)
        if verbose_flag:
            print "Obtained data for ticker %s" % ticker  
    except:
        if verbose_flag:
            print "No data for ticker %s" % ticker  

df = pd.concat(ticker_df_list)            
cell= df[['Ticker','Adj Close']]          
cell.reset_index().sort(['Ticker', 'Date'], ascending=[1,0]).set_index('Ticker')
cell.to_pickle('close_price.pkl')        




if 'cell' not in locals():
    df = pd.read_pickle('close_price.pkl')
else: 
    df = cell 
    
dte1 = '2011-07-01'
dte2 = '2017-01-01'
tickers = sorted(list(set(df['Ticker'].values)))                   
tkrlens = [len(df[df.Ticker==tkr][dte1:dte2]) for tkr in tickers]  
tkrmode = mode(tkrlens)[0][0]                                      


good_tickers = [tickers[i] for i,tkr in enumerate(tkrlens) if tkrlens[i]==tkrmode]  

rtndf = pd.DataFrame()  

 
for tkr in good_tickers: 
    tmpdf = df[df.Ticker==tkr]['Adj Close'][dte1:dte2]
    tmprtndf = ((tmpdf-tmpdf.shift(1))/tmpdf).dropna()
    rsdf = (tmprtndf-tmprtndf.mean())/tmprtndf.std()
    rtndf = pd.concat([rtndf, rsdf],axis=1)

rtndf = rtndf.dropna()
rtndf.columns = good_tickers
t,m = rtndf.shape
cmat = rtndf.corr()                                           
rcmat = (cmat + cmat.T)/2                                    
ncorr = nearcorr(rcmat, max_iterations=1000)                  
ncorrdf = pd.DataFrame(ncorr,columns=good_tickers,index=good_tickers)

sns.clustermap(1-ncorrdf, figsize=(23,23));  

cluster = sch.fclusterdata(1-ncorrdf,0.75,method='average')
tmpgroup = pd.concat([pd.Series(cluster),pd.Series(good_tickers)],axis=1)
groups = dict()
N_cluster = max(cluster)
for k in range(1,N_cluster+1):
    groups[k] = []
    
for i in range(len(cluster)):
    groups[tmpgroup[0][i]].append(tmpgroup[1][i])

Data = pd.DataFrame()

for tkr in good_tickers:
    tmpdf = df[df.Ticker==tkr]['Adj Close'][dte1:dte2]
    Data = pd.concat([Data, tmpdf], axis=1)
Data.columns = good_tickers