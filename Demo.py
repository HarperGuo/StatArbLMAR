# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 04:12:45 2017

@author: Hao Guo
"""
N = 1000
N_out = len(Data) - N
PnL1 = np.zeros(N_out-1)
Freq1 = 0.0
PnL_naive = np.zeros(N_out-1)
Freq_naive = 0.0

for k in range(1,len(groups)+1):
    print k
    zeta = {1:np.array([0.1,0.9]), 2:np.array([0.5,1.5])}
    sigma = {1:np.array([1.0]), 2:np.array([1.0])}
    delta = np.array([-2.0,0.6,0.3])
    if len(groups[k])==1:
        print False
        continue
    else:
        try:            
            Train = np.log(Data[groups[k]][0:N])
            Test = np.log(Data[groups[k]][N:])
            res_ols = OLS(np.log(Train[Train.columns[0]]),\
                           np.log(Train[Train.columns[1:]])).fit_regularized()
            param = np.append(-1.0, res_ols.params)    
            stationary = adftest(res_ols.resid)
            print stationary
            if stationary:
                residuals = np.matrix(Train)*np.matrix(param).T
                residuals_out = np.matrix(Test)*np.matrix(param).T
                residuals = np.array(residuals)
                residuals = residuals.reshape((N,))
                residuals_out = np.array(residuals_out)
                residuals_out = residuals_out.reshape((N_out,))
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
                            print terminal
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
                y1 = np.append(residuals[N-m1+1:],residuals_out)
                x1 = np.zeros(N_out+m1-1)
                y2 = np.append(residuals[N-m2+1:],residuals_out)
                x2 = np.zeros(N_out+m2-1)
                for t1 in range(N_out+m1-1):
                    x1[t1] = Phi1[0] + np.sum(np.flipud(y1[t1:t1+m1-1])*Phi1[1:])
                for t2 in range(N_out+m2-1):
                    x2[t2] = Phi2[0] + np.sum(np.flipud(y2[t2:t2+m2-1])*Phi2[1:])
                x1 = x1[m1-1:]
                x2 = x2[m2-1:]
                mean1 = np.mean(residuals)
                thr = 2.0
                stdev1 = np.std(Var1)
                stdev2 = np.std(Var2)
                sig1 = new_strategy(residuals,N_out,p1_out,p1Lw,p1Up,p2_out,p2Lw,p2Up,mean1,stdev1,mean1,stdev2,thr)
                sig_naive = naive_strategy(residuals,N_out,mean1,stdev1,thr)
                pnl1,freq1 = PnL(sig1,param,Test)
                pnl_naive,freq_naive = PnL(sig_naive, param, Test)
                print pnl1
                PnL1 += pnl1
                print freq1
                Freq1 += freq1
                print pnl_naive
                PnL_naive += pnl_naive   
                print freq_naive
                Freq_naive += freq_naive
            else:
                print "non-stationary"
                continue
        except:
            print "sigular matrix"
            

HitRatio_new = sum(PnL1>=0)*1.0/N_out
HitRatio_naive = sum(PnL_naive>=0)*1.0/N_out
exp_gain_new = sum(PnL1[PnL1>=0])*1.0/sum(PnL1>=0)
exp_gain_naive = sum(PnL_naive[PnL_naive>=0])*1.0/sum(PnL_naive>=0)
exp_loss_new = sum(PnL1[PnL1<0])*1.0/sum(PnL1<0)
exp_loss_naive = sum(PnL_naive[PnL_naive<0])*1.0/sum(PnL_naive<0)

