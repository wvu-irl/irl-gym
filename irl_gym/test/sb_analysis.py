"""
This module contains tests for Stickbug Environment
"""
__license__ = "BSD-3"
__docformat__ = 'reStructuredText'
__author__ = "Trevor Smith"

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from copy import deepcopy
import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from irl_gym.test.file_utils import *

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# def merge (path, num):
#     data = pd.DataFrame()
#     for n in range(num):
#         save_path = path + "_data" + str(n) + ".csv"
#         data = pd.concat([data, import_file(save_path)])
#     export_file(data, path + "_data.csv")
    
# files = ["referee4", "referee6"]
# n = 4

# for file in files:
#     path = current + "/" + file
#     merge(path, n)

# exit()

num = [1,2,4,6]
modes = ["naive", "hungarian", "referee"]
naive_data = pd.DataFrame()
hungarian_data = pd.DataFrame()
# ref_data = pd.DataFrame()
for n in num:
    save_path = current + "/naive" + str(n) + "_data.csv"
    naive_data = pd.concat([naive_data, import_file(save_path)])
    save_path = current + "/hungarian" + str(n) + "_data.csv"
    hungarian_data = pd.concat([hungarian_data, import_file(save_path)])
    save_path = current + "/referee" + str(n) + "_data.csv"
    ref_data = pd.concat([ref_data, pd.read_csv(save_path)])
    
naive_data = naive_data.groupby("num_arms")
hungarian_data = hungarian_data.groupby("num_arms")
ref_data = ref_data.groupby("num_arms")

naive_arms = {}
for name, group in naive_data:
    for arm in num:
        if arm == name:
            naive_arms[arm] = group

hungarian_arms = {}
for name, group in hungarian_data:
    for arm in num:
        if arm == name:
            hungarian_arms[arm] = group

ref_arms = {}
for name, group in ref_data:
   for arm in num:
       if arm == name:
           ref_arms[arm] = group

#group by trial
naive = {}
hungarian = {}
# ref = {}
time = []
for arm in num:
    naive_arms[arm] = naive_arms[arm].groupby("trial")
    hungarian_arms[arm] = hungarian_arms[arm].groupby("trial")
    # ref_arms[arm] = ref_arms[arm].groupby("trial")
    
    naive[arm] = {"n_pol":[], "n_flowers":[]}
    for name, group in naive_arms[arm]:
        time = group.loc[:,"time"].tolist()
        naive[arm]["n_pol"] = group.loc[:,"num_pollinated"].tolist()
        naive[arm]["n_flowers"] = group.loc[:,"num_flowers"].tolist()[0]
        
    hungarian[arm] = {"n_pol":[], "n_flowers":[]}
    for name, group in hungarian_arms[arm]:
        # time = group.loc[:,"time"].tolist()
        hungarian[arm]["n_pol"] = group.loc[:,"num_pollinated"].tolist()
        hungarian[arm]["n_flowers"] = group.loc[:,"num_flowers"].tolist()[0]
        
    
    # ref[arm] = {"n_pol":[], "n_flowers":[]}
    # for name, group in ref_arms[arm]:
    #     # time = group.loc[:,"time"].tolist()
    #     ref[arm]["n_pol"] = group.loc[:,"num_pollinated"].tolist()
    #     ref[arm]["n_flowers"] = group.loc[:,"num_flowers"].tolist()[0]




# time = naive_data[]    
# averages = {}
# for mode in modes:
    


# percentage of flowers pollinated for each set of arms (box.whisker)
# total

# num flowers pollinated for each planner and number of arms
# cdf and rate

#### OTHER -----------------------------
# data_pd = import_file(current + "/test.csv")
# grouped = data_pd.groupby("trial")
# data = []
# for name, group in grouped:
#     datum = {}
#     datum["time"]= group.loc[:,"time"].tolist()
#     datum["n_pol"] = group.loc[:,"num_pollinated"].tolist()
#     datum["n_flowers"] = group.loc[:,"num_flowers"].tolist()[0]
#     data.append(datum)
# # print(data[0])

# t = []
# n = []
# for d in data:
#     t.extend(d["time"])
#     n.extend(d["n_pol"])

# inds = np.unique(np.random.randint(0, len(t), 200))
# # print(inds)
# t = [t[i] for i in inds]
# n = [n[i] for i in inds]

# kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
# gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1, normalize_y=True)

# gpr.fit(np.array(t).reshape(-1,1), np.array(n).reshape(-1,1))

# plt.scatter(t, n, c='r', marker='x', label='Observed data')

# x = np.linspace(0, 1000,100)
# print(x)
# x = (x/10).reshape(-1, 1)
# # print(x)
# y, sigma = gpr.predict(x, return_std=True)

# # Plot the mean prediction
# plt.plot(x,y, 'b-', label='Mean prediction')

# # Plot the confidence interval
# plt.fill_between(x.ravel(), y - 1.96 * sigma, y + 1.96 * sigma, alpha=0.2, color='blue', label='95% Confidence interval')

# plt.xlabel('Time')
# plt.ylabel('n_pollinated')
# plt.title('Gaussian Process Regression')
# plt.legend()

# plt.show()