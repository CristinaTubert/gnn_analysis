
import pandas as pd
import numpy as np
import csv
import os.path
import pickle


import collections
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

import seaborn as sns
  
NAME = 'CoauthorCS'
TYPE_SPLIT = 'random2'
SIZE_SPLIT = '916'

path_in = '../generic_metrics/results/edge_cut/' + NAME + '_' + TYPE_SPLIT + '_' + str(SIZE_SPLIT) + '.csv'
path_in_norm = '../generic_metrics/results/edge_cut/' + NAME + '_' + TYPE_SPLIT + '_' + str(SIZE_SPLIT) + '_norm.csv'
'''
M_norm=[]
with open(path_in_norm,'r') as csvfile:
    values = csv.reader(csvfile, delimiter=',')
    #print(values)
    for row in values:
        #print(row)
        M_norm.append([float(v) for v in row])

plt.figure(dpi=600) 
ax = sns.heatmap(M_norm, annot=True, annot_kws=dict(fontsize=3.5), fmt='.2g')

plt.suptitle('EDGES SHARED RATIO between splits', fontsize=12, weight='bold')
plt.title(title, fontsize=10)
plt.xlabel('split', fontsize=10, labelpad=10.0)
plt.ylabel('split', fontsize=10, labelpad=10.0)
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
ax.tick_params(left=False, bottom=False)

cbar=ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=8)

path_out_norm = './results/edge_cut/' + NAME + '_' + TYPE_SPLIT + '_' + str(SIZE_SPLIT) + '_norm.png'
plt.savefig(path_out_norm)
'''

M=[]
with open(path_in,'r') as csvfile:
    values = csv.reader(csvfile, delimiter=',')
    #print(values)
    for row in values:
        #print(row)
        M.append([float(v) for v in row])

plt.figure(dpi=300)#500 
ax = sns.heatmap(M, annot=True, annot_kws=dict(fontsize=4.5), fmt='g')#2

plt.suptitle('SHARED EDGES between splits', fontsize=10, weight='bold')
if TYPE_SPLIT == 'random2':
    title = NAME + ' [random splits of ' + SIZE_SPLIT + ' nodes]'
else:
    title = NAME + ' [' + TYPE_SPLIT + ' splits of ' + SIZE_SPLIT + ' nodes]'

plt.title(title, fontsize=8)
plt.xlabel('split ID', fontsize=8, labelpad=10.0)
plt.ylabel('split ID', fontsize=8, labelpad=10.0)
plt.xticks(fontsize=6)#5
plt.yticks(fontsize=6)#5
ax.tick_params(left=False, bottom=False)

cbar=ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=8)

path_out = './new_results/edge_cut/' + NAME + '_' + TYPE_SPLIT + '_' + str(SIZE_SPLIT) + '.png'
plt.savefig(path_out)