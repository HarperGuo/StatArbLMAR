# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 10:12:02 2017

@author: Hao Guo
"""
import pickle

path='C:\Users\Hao Guo\Documents\Python Scripts\FRE_StatArb_project'
# write Data
pkl_file_Data = open(path + '\Data.pkl', 'w')
pickle.dump(Data,pkl_file_Data)
# write groups
pkl_file_group = open(path + '\group.pkl', 'w')
pickle.dump(groups,pkl_file_group)
# laod groups
pkl_file_group = open(path + '\group.pkl', 'r')
groups = pickle.load(pkl_file_group)
# load Data
pkl_file_Data = open(path + '\Data.pkl', 'r')
Data = pickle.load(pkl_file_Data)

pkl_file_1k1 = open(path + '\PnL1_k1.pkl', 'w')
pickle.dump(PnL1,pkl_file_1k1)
pkl_file_nk1 = open(path + '\PnL_naive_k1.pkl', 'w')
pickle.dump(PnL_naive,pkl_file_nk1)

pkl_file_1k1 = open(path + '\PnL1_k1.pkl', 'r')
PnL1_k1 = pickle.load(pkl_file_1k1)
pkl_file_nk1 = open(path + '\PnL_naive_k1.pkl', 'r')
PnL_naive_k1 = pickle.load(pkl_file_nk1)

pkl_file_1k2 = open(path + '\PnL1_k2.pkl', 'w')
pickle.dump(PnL1,pkl_file_1k2)
pkl_file_nk2 = open(path + '\PnL_naive_k2.pkl', 'w')
pickle.dump(PnL_naive,pkl_file_nk2)

pkl_file_1k2 = open(path + '\PnL1_k2.pkl', 'r')
PnL1_k2 = pickle.load(pkl_file_1k2)
pkl_file_nk2 = open(path + '\PnL_naive_k1.pkl', 'r')
PnL_naive_k2 = pickle.load(pkl_file_nk2)