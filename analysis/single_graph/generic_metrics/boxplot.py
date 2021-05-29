import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
import seaborn as sns
import sys
from scipy import stats
from collections import Counter
import math
import os, glob

from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

'''
def merge_csv():
    path = '../generic_metrics/results/all/'
    all_files = glob.glob(os.path.join(path, "ogbg-*.csv"))

    df_from_each_file = (pd.read_csv(f, sep=',') for f in all_files)
    df_merged = pd.concat(df_from_each_file, ignore_index=True)
    df_merged.to_csv("merged.csv")
'''

name = 'ogbn-products'

# Import Data
path = '../generic_metrics/results/all/' + name + '.csv'

x = []
y = []
z = []

with open(path,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        if row[0] == 'Dataset name':
            continue
        
        x.append(int(row[3])) #num nodes (split size)
        y.append(float(row[5])) #real avg degree

        if row[-1].find('random') != -1:
            z.append('random')

        else:
            z.append('ordered')

# Draw Plot
plt.figure(figsize=(13,10), dpi= 80)
size_split = sorted(list(set(x)))
ax = sns.boxplot(x=x, y=y, hue=z, order=size_split, notch=False, linewidth=0.5, fliersize=3, palette='Set2')
plt.setp(ax.lines, color='red')
for i in range(len(size_split)-1):
    plt.vlines(i+.5, max(y), min(y), linestyles='solid', colors='gray', alpha=0.2)


# Decoration
plt.xticks(rotation=45)
plt.title('Box Plot of ogbn-products real average by split size', y=1.05, fontsize=22)

plt.grid(axis='y', color='gray')
sns.despine()
path_out = './realavgdegree' + name + '.png'
plt.savefig(path_out)

'''
path = '../generic_metrics/results/all/' + name + '.csv'
with open(path,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        if row[0] == 'Dataset name':
            continue
        x.append(int(row[5]))
        y.append(int(row[6]))

xy = list(zip(x,y))
cnt = Counter(xy)
z = [cnt[coord] for coord in xy]

plt.scatter(x,y,s=2,c=z,cmap='gnuplot')

plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

#plt.yticks(np.arange(min(y), max(y)+1, step=100))
plt.xlabel('# nodes')
plt.ylabel('# edges')
title = 'ogbg-' + name + ' [' + str(len(x)) + ' graphs]'
plt.suptitle('NODE-EDGE CORRELATION', y=0.98, fontsize=10, fontweight='bold')
plt.title(title, fontsize=9)
#plt.legend(fontsize=8, loc='best', markerscale=2.0)

cbar = plt.colorbar()
#t = 5*round((len(x)//10)/5)
#cbar.set_ticks(MultipleLocator(t))
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_title('# graphs', fontsize=8)

plt.grid()

path_out = './node-edge/ogbg-' + name + '.png'
plt.savefig(path_out)

'''