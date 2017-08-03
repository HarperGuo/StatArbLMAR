# -*- coding: utf-8 -*-
"""
Created on Thu Mar 02 20:16:02 2017

@author: Hao Guo
"""
import numpy as np
#import scipy.optimize as opt
import scipy.stats as dist
import math

# initialization
zeta = {1:np.array([0.1,0.9]), 2:np.array([0.5,1.5])}
sigma = {1:np.array([1.0]), 2:np.array([1.0])}
delta = np.array([-2.0,0.6,0.3])
m1 = len(zeta[1])
m2 = len(zeta[2])
n = len(delta)
s = max([m1,m2,n])

def calcR(Delta, residuals):
    n = len(Delta)
    r = np.zeros(len(residuals)-n+1)
    for t in range(len(residuals)-n+1):
        r[t] = Delta[0] + np.sum(np.abs(np.flipud(residuals[t:t+n-1])) * Delta[1::])
    return r

def calcP(Delta, residuals):
    r = calcR(Delta, residuals)
    er = np.exp(r)
    p = er/(1+er)
    return p
    
def calcE(phi, residuals):
    m = len(phi)
    e = np.zeros(len(residuals)-m+1)
    for t in range(len(residuals)-m+1):
        e[t] = residuals[t+m-1] - phi[0] - np.sum(np.flipud(residuals[t:t+m-1])*phi[1::])
    return e
    
def calcElist(phi1, phi2, residuals):
    E = {1:calcE(phi1, residuals), 2:calcE(phi2, residuals)}
    return E
    
def calcZ(E, p, sigma): 
    global m1
    global m2
    global n
    lag1 = max([m1,n])-min([m1,n])
    lag2 = max([m2,n])-min([m2,n])
    if n>=m1:
        d1 = p/(sigma[0]*dist.norm.pdf(E[1],loc=0,scale=sigma[0]))[lag1::]
    else:
        d1 = p[lag1::]/(sigma[0]*dist.norm.pdf(E[1],loc=0,scale=sigma[0]))
    if n>=m2:
        d2 = (1-p)/sigma[1]*dist.norm.pdf(E[2],loc=0,scale=sigma[1])[lag2::]
    else:
        d2 = (1-p)[lag2::]/sigma[1]*dist.norm.pdf(E[2],loc=0,scale=sigma[1])
    Z = {1: d1/(d1+d2), 2: d2/(d1+d2)}
    return Z



def calcL(E, p, sigma, Z, s):
    l1 = np.sum(Z[1][s::]*np.log(p[s::])\
                  - 0.5*np.log(math.pow(sigma[0],2)) \
                  - E[1][s::]*E[1][s::]/(2.0*math.pow(sigma[0],2)))
    l2 = np.sum(Z[2][s::]*np.log(p[s::])\
                  - 0.5*np.log(math.pow(sigma[1],2)) \
                  - E[2][s::]*E[2][s::]/(2.0*math.pow(sigma[1],2)))
    log_lik = l1 + l2
    return log_lik


def calcLPt1(Delta):
    global residuals
    global s
    global Z
    p = calcP(Delta, residuals)
    lstar = np.sum(Z[1]*np.log(p)+Z[2]*np.log(1.0-p))
    return lstar

def calcEpsilon(residuals, Delta, Z, n):
    calcP(Delta, residuals)
    sum1 = np.matrix(np.zeros((n+1,1)))
    sum2 = np.matrix(np.zeros((n+1,n+1)))
    for t in range(len(residuals)-n):
        delRdelEp = np.append(1.0, np.abs(np.flipud(residuals[t:t+n])))
        sum1 += (Z[1][t]-p[t])*np.matrix(delRdelEp).T
        sum2 += (p[t])*(1.0-p[t])*(-1.0)* np.matrix(delRdelEp).T * np.matrix(delRdelEp)
    return Delta-np.array(np.linalg.inv(sum2)*(sum1))
    
def calcZeta(residuals, Z, k):
    global m1
    global m2
    global s
    if k == 1:
        m = m1
    elif k == 2:
        m = m2
    sum1 = np.matrix(np.zeros((m,m)))
    sum2 = np.matrix(np.zeros((m,1)))
    for t in range(len(residuals)-s):
        delEdelZeta = np.append(1,np.flipud(residuals[t+s-m:t+s-1]))*(-1.0)        
        sum1 += Z[k][t]*np.matrix(delEdelZeta).T * np.matrix(delEdelZeta)
        sum2 += Z[k][t]*residuals[t+s]*np.matrix(delEdelZeta).T
    return -np.array(np.linalg.inv(sum1)*(sum2))

def calcSigma(residuals, Z, s, k):
    global m1
    global m2
    if k == 1:
        m = m1
    elif k == 2:
        m = m2
    sum2 = 0.0
    zeta = calcZeta(residuals, Z, k)
    sz = zeta.size
    zeta = zeta.reshape((sz,))
    res = residuals - np.average(residuals)
    for t in range(len(residuals)-s):
        delEdelZeta = np.append(1,np.flipud(residuals[t+s-m:t+s-1]))*(-1.0)   
        #temp = np.matrix(zeta).T*np.matrix(delEdelZeta)
        sum2 += Z[k][t] * (res[t+s]+np.dot(delEdelZeta,zeta))
    sum1 = np.sum(Z[k])
    return sum2/sum1

def checkConverge(t, zeta, sigma, Delta, m1, m2, n, threshold):
    if t == 0:
        return False
    elif any(np.abs(zeta[1][t*m1:(t+1)*m1]/zeta[1][(t-1)*m1:t*m1] - 1.0) > threshold):
        return False
    elif any(np.abs(zeta[2][t*m2:(t+1)*m2]/zeta[2][(t-1)*m2:t*m2] - 1.0) > threshold):
        return False
    elif np.abs(sigma[1][t]/sigma[1][t-1] - 1.0) > threshold:
        return False
    elif np.abs(sigma[2][t]/sigma[2][t-1] - 1.0) > threshold:
        return False
    elif any(np.abs(Delta[t*n:(t+1)*n]/Delta[(t-1)*n:t*n]-1) > threshold):
        return False
    return True
        

threshold = 1e-5
""""
for itr in range(100):
    p = calcP(delta[itr*n:(itr+1)*n],residuals)
    E = calcElist(zeta[1][itr*m1:(itr+1)*m1], zeta[2][itr*m2:(itr+1)*m2], residuals)
    Sigma=[sigma[1][itr],sigma[2][itr]]
    Z = calcZ(E, p, Sigma)
    zeta[1] = np.append(zeta[1],calcZeta(residuals,Z,1).T)
    zeta[2] = np.append(zeta[2],calcZeta(residuals,Z,2).T)
    sigma[1] = np.append(sigma[1],calcSigma(residuals,Z,s,1))
    sigma[2] = np.append(sigma[2],calcSigma(residuals,Z,s,2))
    delta = np.append(delta, calcEpsilon(residuals, delta[itr*n:(itr+1)*n], Z, n))
    #delta_min = opt.minimize(calcLPt1, delta[itr*n:(itr+1)*n],  method='Nelder-Mead', tol=1e-5)    
    #delta = np.append(delta, delta_min)
    if itr == 0:
        continue
    else:
        if checkConverge(itr, zeta, sigma, delta, m1, m2, n, threshold) == True:
            terminal = itr
            break
        
zeta[1] = zeta[1][~np.isnan(zeta[1])]
zeta[2] = zeta[2][~np.isnan(zeta[2])]
Phi1 = zeta[1][len(zeta[1])-m1:]
Phi2 = zeta[2][len(zeta[2])-m2:]
sigma[1] = sigma[1][~np.isnan(sigma[1])]
sigma[2] = sigma[2][~np.isnan(sigma[2])]
Var1 = sigma[1][-1]
Var2 = sigma[2][-1]
delta = delta[~np.isnan(delta)]
Delta = delta[len(delta)-n:]
"""