# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 19:04:05 2017

@author: Hao Guo
"""

import matplotlib.pyplot as plt 

fig1 = plt.figure(figsize=(10,3))
plt.title('PnL when k=1')
ax1 = fig1.add_subplot(1,2,1)
ax1.plot(np.cumsum(PnL1_k1))
ax1.set_xlabel('New Strategy')
ax2 = fig1.add_subplot(1,2,2)
ax2.plot(np.cumsum(PnL_naive_k1))
ax2.set_xlabel('Naive Strategy')

fig2 = plt.figure(figsize=(10,3))
plt.title('PnL when k=2')
ax1 = fig2.add_subplot(1,2,1)
ax1.plot(np.cumsum(PnL1))
ax1.set_xlabel('New Strategy')
ax2 = fig2.add_subplot(1,2,2)
ax2.plot(np.cumsum(PnL_naive))
ax2.set_xlabel('Naive Strategy')

fig3 = plt.figure(figsize=(10,10))
plt.title('residuals with two states')
ax1 = fig3.add_subplot(2,1,1)
ax1.plot(residuals_out,label='residual')
ax1.plot(x1,'k-',label='state 1')
ax1.plot(x2,'g-',label='state 2')
ax1.legend(loc='best')
ax2 = fig3.add_subplot(2,1,2)
ax2.plot(p1_out,'k-',label='p1')
ax2.plot(p2_out,'g-',label='p2')
ax2.legend(loc='best')