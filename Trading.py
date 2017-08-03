# -*- coding: utf-8 -*-
"""
Created on Thu Mar 09 21:37:17 2017

@author: Hao Guo
"""
import numpy as np
"""
p1 = calcP(Delta, np.append(residuals,residuals_out))
p2 = 1.0 - p1
p1_in = p1[:N-n+1]
p2_in = p2[:N-n]
p1_out = p1[N-n+1:]
p2_out = p2[N-n+1:]

p1Lw = np.percentile(p1_in,50)
p1Up = np.percentile(p1_in,95)
p2Lw = np.percentile(p2_in,50)
p2Up = np.percentile(p2_in,95)
"""
"""
err1 = calcE(Phi1, np.append(residuals,residuals_out))
err2 = calcE(Phi2, np.append(residuals,residuals_out))
err1_out = err1[N-m1:]
err2_out = err2[N-m2:]
"""
def new_strategy(residuals,N_out,p1_out,p1Lw,p1Up,p2_out,p2Lw,p2Up,x1,stdev1,x2,stdev2,k):
    sig1 = np.zeros(N_out)
    for t in range(N_out-1):
        if p1_out[t+1] >= p2_out[t+1]:
            if p1_out[t+1]>=p1Up and residuals[t+1]>(x1+k*stdev1) and residuals[t]<(x1+k*stdev1) \
                                                 and sig1[t]==0:
                 sig1[t+1] = -1
            elif residuals[t+1]<x1 and residuals[t]>x1 and sig1[t]==-1:                                              
                 sig1[t+1] = 0
            elif p1_out[t+1]>=p1Up and residuals[t+1]<(x1-k*stdev1) and residuals[t]>(x1-k*stdev1) \
                                         and sig1[t]==0:
                 sig1[t+1] = 1
            elif residuals[t+1]>x1 and residuals[t]<x1 and sig1[t]==1:
                 sig1[t+1] = 0
            else:
                 sig1[t+1] = sig1[t]
        elif p1_out[t+1] < p2_out[t+1]:
            if p2_out[t+1]>=p2Up and residuals[t+1]>(x2+k*stdev2) and residuals[t]<(x2+k*stdev2) \
                                                and sig1[t]==0:
                 sig1[t+1] = -1
            elif residuals[t+1]<x2 and residuals[t]>x2 and sig1[t]==-1:                                              
                 sig1[t+1] = 0
            elif p1_out[t+1]>=p2Up and residuals[t+1]<(x2-k*stdev2) and residuals[t]>(x2-k*stdev2) \
                                         and sig1[t]==0:
                 sig1[t+1] = 1
            elif residuals[t+1]>x2 and residuals[t]<x2 and sig1[t]==1:
                 sig1[t+1] = 0
            else:
                 sig1[t+1] = sig1[t]
    return sig1
"""
def new_strategy(residuals,N_out,p1_out,p1Lw,p1Up,p2_out,p2Lw,p2Up,x1,stdev1,x2,stdev2,k):
    sig1 = np.zeros(N_out)
    for t in range(N_out-1):
        if p1_out[t+1] >= p2_out[t+1]:
            if p1_out[t+1]>=p1Up and residuals[t+1]>(x1[t+1]+k*stdev1) and residuals[t]<(x1[t]+k*stdev1) \
                                                 and sig1[t]==0:
                 sig1[t+1] = -1
            elif residuals[t+1]<x1[t+1] and residuals[t]>x1[t] and sig1[t]==-1:                                              
                 sig1[t+1] = 0
            elif p1_out[t+1]>=p1Up and residuals[t+1]<(x1[t+1]-k*stdev1) and residuals[t]>(x1[t]-k*stdev1) \
                                         and sig1[t]==0:
                 sig1[t+1] = 1
            elif residuals[t+1]>x1[t+1] and residuals[t]<x1[t] and sig1[t]==1:
                 sig1[t+1] = 0
            else:
                 sig1[t+1] = sig1[t]
        elif p1_out[t+1] < p2_out[t+1]:
            if p2_out[t+1]>=p2Up and residuals[t+1]>(x2[t+1]+k*stdev2) and residuals[t]<(x2[t]+k*stdev2) \
                                                and sig1[t]==0:
                 sig1[t+1] = -1
            elif residuals[t+1]<x2[t+1] and residuals[t]>x2[t] and sig1[t]==-1:                                              
                 sig1[t+1] = 0
            elif p1_out[t+1]>=p2Up and residuals[t+1]<(x2[t+1]-k*stdev2) and residuals[t]>(x2[t]-k*stdev2) \
                                         and sig1[t]==0:
                 sig1[t+1] = 1
            elif residuals[t+1]>x2[t+1] and residuals[t]<x2[t] and sig1[t]==1:
                 sig1[t+1] = 0
            else:
                 sig1[t+1] = sig1[t]
    return sig1
"""        
def naive_strategy(residuals,N_out,mean1,stdev1,k):
    sig_naive = np.zeros(N_out)
    for t in range(N_out-1):
        if residuals[t+1]>(mean1+k*stdev1) and residuals[t]<(mean1+k*stdev1) \
                                             and sig_naive[t]==0:
            sig_naive[t+1] = -1
        elif residuals[t+1]<mean1 and residuals[t]>mean1 \
                                               and sig_naive[t]==-1:
            sig_naive[t+1] = 0
        elif residuals[t+1]<(mean1-k*stdev1) and residuals[t]>(mean1-k*stdev1) \
                                         and sig_naive[t]==0:
            sig_naive[t+1] = 1
        elif residuals[t+1]>mean1 and residuals[t]<mean1 \
                                              and sig_naive[t]==1:
            sig_naive[t+1] = 0
        else:
            sig_naive[t+1] = sig_naive[t]
    return sig_naive

"""


for t in range(N_out-1):
    if y1[t+1]>(x1[t+1]+k*stdev1) and y1[t]<(x1[t]+k*stdev1) \
                                         and sig1[t]==0 and p1_out[t+1]>=p1Up:
        sig1[t+1] = -1
    elif y1[t+1]<x1[t+1] and y1[t]>x1[t] and sig1[t]==-1 and p1_out[t+1]<=p1Lw:
        sig1[t+1] = 0
    elif y1[t+1]<(x1[t+1]-k*stdev1) and y1[t]>(x1[t]-k*stdev1) \
                                         and sig1[t]==0 and p1_out[t+1]>=p1Up:
        sig1[t+1] = 1
    elif y1[t+1]>x1[t+1] and y1[t]<x1[t] and sig1[t]==1 and p1_out[t+1]<=p1Lw:
        sig1[t+1] = 0
    else:
        sig1[t+1] = sig1[t]
"""
def PnL(sig,param,price):
    holdings = np.matrix(sig).T * np.matrix(param)
    tmp = np.array(holdings[1:,:]) * (price.values[1:,:]-price.values[0:-1,:]) 
    pnl = np.sum(tmp,axis=1)
    freq = np.round(np.sum(np.diff(sig)<>0)/2.0)
    return pnl,freq