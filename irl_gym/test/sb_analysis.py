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
#         save_path = path + "_data" + str(n+1) + ".csv"
#         data = pd.concat([data, import_file(save_path)])
#     export_file(data, path + "_data.csv")
    
# files = ["referee4", "referee6"]
# n = 4

# for file in files:
#     path = current + "/" + file
#     merge(path, n)

# exit()

# files = ["hungarian6", "naive6"]
# for file in files:
#     save_path = current + "/" + file + "_data.csv"
#     data = import_file(save_path)
#     data["num_arms"] = 6
    
#     save_path = current + "/" + file + "_data_new.csv"
#     export_file(data, save_path)

# exit()

num = [1,2,4,6]
modes = ["naive", "hungarian", "referee"]
naive_data = pd.DataFrame()
hungarian_data = pd.DataFrame()
ref_data = pd.DataFrame()
for n in num:
    save_path = current + "/naive" + str(n) + "_data.csv"
    naive_data = pd.concat([naive_data, import_file(save_path)])
    save_path = current + "/hungarian" + str(n) + "_data.csv"
    hungarian_data = pd.concat([hungarian_data, import_file(save_path)])
    save_path = current + "/referee" + str(n) + "_data.csv"
    ref_data = pd.concat([ref_data, pd.read_csv(save_path)])
    
naive_data = naive_data.groupby("num_arms")
hungarian_data = hungarian_data.groupby("num_arms")
referee_data = ref_data.groupby("num_arms")

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
for name, group in referee_data:
    for arm in num:
        if arm == name:
            ref_arms[arm] = group

#group by trial
naive = {}
hungarian = {}
ref = {}
time = []
for arm in num:
    naive_arms[arm] = naive_arms[arm].groupby("trial")
    hungarian_arms[arm] = hungarian_arms[arm].groupby("trial")
    ref_arms[arm] = ref_arms[arm].groupby("trial")
    
    naive[arm] = {"n_pol":[], "n_flowers":[]}
    for name, group in naive_arms[arm]:
        time = group.loc[:,"time"].tolist()
        naive[arm]["n_pol"].append(group.loc[:,"num_pollinated"].tolist())
        naive[arm]["n_flowers"].append(group.loc[:,"num_flowers"].tolist()[0])
        
    hungarian[arm] = {"n_pol":[], "n_flowers":[]}
    for name, group in hungarian_arms[arm]:
        # time = group.loc[:,"time"].tolist()
        hungarian[arm]["n_pol"].append(group.loc[:,"num_pollinated"].tolist())
        hungarian[arm]["n_flowers"].append(group.loc[:,"num_flowers"].tolist()[0])
        
    ref[arm] = {"n_pol":[], "n_flowers":[], "inter_arm_conflicts":[], "flower_assignment_conflicts":[], "no_flowers_conflicts":[], "total_conflicts":[]}
    for name, group in ref_arms[arm]:
        # time = group.loc[:,"time"].tolist()
        ref[arm]["n_pol"].append(group.loc[:,"num_pollinated"].tolist())
        ref[arm]["n_flowers"].append(group.loc[:,"num_flowers"].tolist()[0])
        ref[arm]["inter_arm_conflicts"].append(group.loc[:,"inter_arm_conflicts"].tolist()[0])
        ref[arm]["flower_assignment_conflicts"].append(group.loc[:,"flower_assignment_conflicts"].tolist()[0])
        ref[arm]["no_flowers_conflicts"].append(group.loc[:,"no_flowers_conflicts"].tolist()[0])
        ref[arm]["total_conflicts"].append(group.loc[:,"total_conflicts"].tolist()[0])

#box and whisker plot for arms and mode
naive_total_pol = {}
hungarian_total_pol = {}
referee_total_pol = {}

for arm in num:
    
    naive_total_pol[arm] = []
    for i in range(len(naive[arm]["n_pol"])):
        naive_total_pol[arm].append(naive[arm]["n_pol"][i][-1])
    
    hungarian_total_pol[arm] = []
    for i in range(len(hungarian[arm]["n_pol"])):
        hungarian_total_pol[arm].append(hungarian[arm]["n_pol"][i][-1])
    
    referee_total_pol[arm] = []
    for i in range(len(ref[arm]["n_pol"])):
        referee_total_pol[arm].append(ref[arm]["n_pol"][i][-1])
    
naive_pol_stats = {}
hungarian_pol_stats = {}
referee_pol_stats = {}

legend_list = []
data_list = []

for arm in num:
    naive_pol_stats[arm] = {"mean":np.mean(naive_total_pol[arm]), "std":np.std(naive_total_pol[arm])}
    hungarian_pol_stats[arm] = {"mean":np.mean(hungarian_total_pol[arm]), "std":np.std(hungarian_total_pol[arm])}
    referee_pol_stats[arm] = {"mean":np.mean(referee_total_pol[arm]), "std":np.std(referee_total_pol[arm])}
    
    naive_pol_stats[arm]["min"] = np.min(naive_total_pol[arm])
    naive_pol_stats[arm]["max"] = np.max(naive_total_pol[arm])
    hungarian_pol_stats[arm]["min"] = np.min(hungarian_total_pol[arm])
    hungarian_pol_stats[arm]["max"] = np.max(hungarian_total_pol[arm])
    referee_pol_stats[arm]["min"] = np.min(referee_total_pol[arm])
    referee_pol_stats[arm]["max"] = np.max(referee_total_pol[arm])
    
    naive_pol_stats[arm]["q1"] = np.percentile(naive_total_pol[arm], 25)
    naive_pol_stats[arm]["q3"] = np.percentile(naive_total_pol[arm], 75)
    hungarian_pol_stats[arm]["q1"] = np.percentile(hungarian_total_pol[arm], 25)
    hungarian_pol_stats[arm]["q3"] = np.percentile(hungarian_total_pol[arm], 75)
    referee_pol_stats[arm]["q1"] = np.percentile(referee_total_pol[arm], 25)
    referee_pol_stats[arm]["q3"] = np.percentile(referee_total_pol[arm], 75)
    
    legend_list.append("naive" + str(arm))
    legend_list.append("hungarian" + str(arm))
    legend_list.append("referee" + str(arm))
    
    data_list.append(naive_total_pol[arm])
    data_list.append(hungarian_total_pol[arm])
    data_list.append(referee_total_pol[arm])


## AVG TOTAL POLLINATED -----------------------------
f, ax = plt.subplots()
bp =ax.boxplot(data_list, patch_artist=True)

fn = plt.get_fignums()
# lenged_list = [""] + legend_list
# plt.xticks(range(len(legend_list)),legend_list)
plt.setp(bp['boxes'], color='black')
plt.setp(bp['whiskers'], color='black')
plt.setp(bp['fliers'], color='red', marker='+')

ax.set_xticklabels(legend_list,rotation =45)

colors = ['blue', 'green', 'red']
#set the colors of the boxes
for i in range(len(legend_list)):
    plt.setp(bp['boxes'][i], color=colors[i%3])
    plt.setp(bp["boxes"][i], facecolor = colors[i%3])

plt.savefig(current + "/total_pollinated_box.png")

## AVG RELATIVE POLLINATED -----------------------------
naive_rel_pol = {}
hungarian_rel_pol = {}
referee_rel_pol = {}

data_list = []
for arm in num:
    naive_rel_pol[arm] = np.array(naive_total_pol[arm])/np.array(naive[arm]["n_flowers"])
    hungarian_rel_pol[arm] = np.array(hungarian_total_pol[arm])/np.array(hungarian[arm]["n_flowers"])
    referee_rel_pol[arm] = np.array(referee_total_pol[arm])/np.array(ref[arm]["n_flowers"])
    
    data_list.append(naive_rel_pol[arm])
    data_list.append(hungarian_rel_pol[arm])
    data_list.append(referee_rel_pol[arm])
    
f, ax = plt.subplots()
bp =ax.boxplot(data_list, patch_artist=True)

fn = plt.get_fignums()
# lenged_list = [""] + legend_list
# plt.xticks(range(len(legend_list)),legend_list)
plt.setp(bp['boxes'], color='black')
plt.setp(bp['whiskers'], color='black')
plt.setp(bp['fliers'], color='red', marker='+')

ax.set_xticklabels(legend_list,rotation =45)

colors = ['blue', 'green', 'red']
#set the colors of the boxes
for i in range(len(legend_list)):
    plt.setp(bp['boxes'][i], color=colors[i%3])
    plt.setp(bp["boxes"][i], facecolor = colors[i%3])

plt.savefig(current + "/rel_pollinated_box.png")

## POL OVER TIME -----------------------------
filter_size = 10

naive_avg_pol_sequence = {}
hungarian_avg_pol_sequence = {}
referee_avg_pol_sequence = {}
# legend_list = []
for arm in num:
    naive_seq = np.zeros(len(naive[arm]["n_pol"][0]))
    hungarian_seq = np.zeros(len(hungarian[arm]["n_pol"][0]))
    referee_seq = np.zeros(len(ref[arm]["n_pol"][0]))

    for i in range(len(naive[arm]["n_pol"])):
        naive_seq += np.array(naive[arm]["n_pol"][i])
        hungarian_seq += np.array(hungarian[arm]["n_pol"][i])
        referee_seq += np.array(ref[arm]["n_pol"][i])
        
    temp = deepcopy(naive_seq)
    for i in range(len(naive_seq)-1):
        temp[i] = naive_seq[i+1]-naive_seq[i]
    naive_seq = temp[0:-1]
    
    temp = deepcopy(hungarian_seq)
    for i in range(len(hungarian_seq)-1):
        temp[i] = hungarian_seq[i+1]-hungarian_seq[i]
    hungarian_seq = temp[0:-1]
    
    temp = deepcopy(referee_seq)
    for i in range(len(referee_seq)-1):
        temp[i] = referee_seq[i+1]-referee_seq[i]
    referee_seq = temp[0:-1]
    
    
    naive_avg_pol_sequence[arm] = naive_seq/len(naive[arm]["n_pol"])
    hungarian_avg_pol_sequence[arm] = hungarian_seq/len(hungarian[arm]["n_pol"])
    referee_avg_pol_sequence[arm] = referee_seq/len(ref[arm]["n_pol"])
        
f, ax = plt.subplots()

for arm in num:
    plt.plot(naive_avg_pol_sequence[arm], label="naive" + str(arm))
    plt.plot(hungarian_avg_pol_sequence[arm], label="hungarian" + str(arm))
    plt.plot(referee_avg_pol_sequence[arm], label="referee" + str(arm))

plt.legend()
plt.savefig(current + "/pol_over_time.png")


for arm in num:
    f = plt.figure()
    naive_avg_pol_sequence[arm] = np.convolve(naive_avg_pol_sequence[arm], np.ones((filter_size,))/filter_size, mode='valid')
    hungarian_avg_pol_sequence[arm] = np.convolve(hungarian_avg_pol_sequence[arm], np.ones((filter_size,))/filter_size, mode='valid')
    referee_avg_pol_sequence[arm] = np.convolve(referee_avg_pol_sequence[arm], np.ones((filter_size,))/filter_size, mode='valid')
    
    plt.plot(naive_avg_pol_sequence[arm], label="naive" + str(arm))
    plt.plot(hungarian_avg_pol_sequence[arm], label="hungarian" + str(arm))
    plt.plot(referee_avg_pol_sequence[arm], label="referee" + str(arm))
    
    plt.title = str(arm) + " arms"
    plt.legend()
    plt.savefig(current + "/pol_over_time_filtered"+str(arm)+".png")
    
f = plt.figure()
for arm in num:
    plt.plot(naive_avg_pol_sequence[arm], label=str(arm))
plt.title = "naive"
plt.legend()
plt.savefig(current + "/naive_pol_over_time_filtered.png")

f = plt.figure()
for arm in num:
    plt.plot(hungarian_avg_pol_sequence[arm], label=str(arm))
plt.title = "hungarian"
plt.legend()
plt.savefig(current + "/hungarian_pol_over_time_filtered.png")

f = plt.figure()
for arm in num:
    plt.plot(referee_avg_pol_sequence[arm], label=str(arm))
plt.title = "referee"
plt.legend()
plt.savefig(current + "/referee_pol_over_time_filtered.png")


## Conflicts -----------------------------
f = plt.figure()
data_list = []
for arm in num:
    data_list.append(ref[arm]["inter_arm_conflicts"])
    
f, ax = plt.subplots()
bp =ax.boxplot(data_list, patch_artist=True, whis=10)
ax.set_xticklabels(num)
plt.savefig(current + "/inter_arm_conflict.png")

f = plt.figure()
data_list = []
for arm in num:
    data_list.append(ref[arm]["flower_assignment_conflicts"])

f, ax = plt.subplots()
bp =ax.boxplot(data_list, patch_artist=True, whis=2)
ax.set_xticklabels(num)
plt.savefig(current + "/flower_assignment_conflict.png")

f = plt.figure()
data_list = []
for arm in num:
    data_list.append(ref[arm]["no_flowers_conflicts"])

f, ax = plt.subplots()
bp = ax.boxplot(data_list, patch_artist=True, whis=4)
ax.set_xticklabels(num)
plt.savefig(current + "/no_flowers_conflict.png")