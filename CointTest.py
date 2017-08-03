# -*- coding: utf-8 -*-
"""
Created on Mon Mar 06 10:28:13 2017

@author: Hao Guo
"""
import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import adfuller



def adftest(v, crit='5%', max_d=6, reg='nc', autolag='AIC'):
    boolean = False    
    adf = adfuller(v, max_d, reg, autolag)
    if(adf[0] < adf[4][crit]):
        pass
    else:
        boolean = True
    return boolean
""" 
res_ols = OLS(np.log(Train[Train.columns[0]]),np.log(Train[Train.columns[1:]])).fit_regularized()
param = np.append(-1.0, res_ols.params)    
stationary = adftest(res_ols.resid)
if stationary:
    residuals = np.matrix(Train)*np.matrix(param).T
    residuals_out = np.matrix(Test)*np.matrix(param).T
    N_out = residuals_out.size
    residuals = np.array(residuals)
    residuals = residuals.reshape((N,))
    residuals_out = np.array(residuals_out)
    residuals_out = residuals_out.reshape((N_out,))
"""