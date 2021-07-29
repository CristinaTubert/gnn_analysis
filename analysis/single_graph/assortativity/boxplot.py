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
from matplotlib.cbook import boxplot_stats
from itertools import groupby
from operator import itemgetter

from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import matplotlib as mpl
import matplotlib.patheffects as path_effects

'''
def merge_csv():
    path = '../generic_metrics/results/all/'
    all_files = glob.glob(os.path.join(path, "ogbg-*.csv"))

    df_from_each_file = (pd.read_csv(f, sep=',') for f in all_files)
    df_merged = pd.concat(df_from_each_file, ignore_index=True)
    df_merged.to_csv("merged.csv")
'''

name = 'ogbn-products'
yname = 'homophily ratio'
num_nodes = 2449029

# Import Data
path = './results/all/' + name + '.csv'

x = []
y = []
z = []

with open(path,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        if row[0] == 'Dataset name':
            continue

        x.append(int(row[4])) #num nodes (split size)
        homophily = float(row[6])
        y.append(homophily)

        if row[-1].find('random2') != -1:
            z.append('random')

        else:
            z.append('ordered')

        if int(row[4]) == num_nodes:
            Homophily = float(row[6])

# Draw Plot
plt.figure(figsize=(13,10), dpi= 300)
size_split = sorted(list(set(x)))
colors={'ordered':'cornflowerblue', 'random':'lightpink'}
ax = sns.boxplot(x=x, y=y, hue=z, order=size_split, notch=False, linewidth=0.5, fliersize=3, palette=colors, meanline=True, whis=1.5, flierprops=dict(markerfacecolor='purple', markeredgecolor='purple'))
plt.setp(ax.lines, color='red')
for i in range(len(size_split)-1):
    plt.vlines(i+.5, max(y)+0.003, min(y)-0.001, linestyles='dashed', colors='gray', alpha=0.5)


xz = [(x[i], z[i]) for i in range(0, len(x))]
result = {i:[] for i in xz}
for i,j in zip(xz, y):
    result[i].append(j)

all_outliers = []
for size,values in result.items():
    outliers=[]
    q1,q3 = np.percentile(values, [25,75])
    iqr = q3-q1
    lower_range = q1-(1.5*iqr)
    upper_range = q3+(1.5*iqr)
    for v in values:
        if v < lower_range or v > upper_range:
            outliers.append(v)
    all_outliers.append((size,len(outliers)/len(values)))
all_outliers = sorted(all_outliers)

sizes = sorted(list(set([s[0] for s,o in all_outliers])))
xticks = ax.get_xticks()


print(sizes)
outlier_i = 0
tick = 0
while outlier_i < len(all_outliers) and tick < len(xticks):
    outlier=all_outliers[outlier_i]

    if (outlier[0][0] == sizes[tick] and outlier[0][1] == 'ordered'):
        ax.text(xticks[tick]-0.25, max(y)+0.008, '{:.4f}'.format(outlier[1]), horizontalalignment='center', size='x-small', color='purple', rotation=45)
        outlier_i+=1

    elif (outlier[0][0] == sizes[tick] and outlier[0][1] == 'random'):
        ax.text(xticks[tick]+0.25, max(y)+0.008, '{:.4f}'.format(outlier[1]), horizontalalignment='center', size='x-small', color='purple', rotation=45)
        outlier_i+=1

    else:
        tick+=1

ax.text(-0.9, max(y)+0.0093, 'outliers \nratio', horizontalalignment='center', size='x-small', color='purple')


annotate = str('{:.4f}'.format(Homophily))
ax.annotate(annotate, (xticks[-1]-0.25, Homophily), xytext=(xticks[-1]-0.25, Homophily+0.002), xycoords='data', textcoords='data',  arrowprops=dict(alpha=0.1, arrowstyle='wedge, tail_width=0.5', color='red'), bbox=dict(boxstyle='round', alpha=0.1, color='red'), size=11, ha='left', va='center')


'''
lines = ax.axes.get_lines()
boxes = [c for c in ax.get_children() if type(c).__name__ == 'PathPatch']
lines_per_box = int(len(lines)/len(boxes))

count = 0.0
for median in lines[2:len(lines):lines_per_box]:
    x,y = (data.mean() for data in median.get_data())
    value = x if (median.get_xdata()[1]-median.get_xdata()[0]) == 0 else y
    text = ax.text(x,y, '{:.3f}'.format(value), ha='center', va='center_baseline', color='white', fontsize=8)
    text.set_path_effects([path_effects.Stroke(linewidth=1, foreground='red'), path_effects.Normal(),])
    count+=0.005
'''
# Decoration
#y_major = mpl.ticker.LogLocator(base=10.0, numticks=5)
#ax.yaxis.set_major_locator(y_major)

plt.xlabel('# nodes', fontsize=12, labelpad=18.0)
plt.ylabel(yname, fontsize=12, labelpad=20.0)

xticklabels=[]
for s in sizes:
    l = str(s)+ '\n ('+ str('{:.2f}'.format((s/num_nodes) * 100)) + '%)'
    xticklabels.append(l)

plt.xticks(ticks=xticks, labels=xticklabels)
plt.ylim(min(y)-0.008, max(y)+0.008)

plt.subplots_adjust(bottom=0.1, top=0.85)
suptitle = yname.upper() + ' by split size'
title = name + ' [' + str(num_nodes) + ' nodes]'
plt.suptitle(suptitle, y=0.95, fontsize=15, fontweight='bold')
plt.title(title, fontsize=12, pad=30.0)

plt.legend(loc='lower left')
plt.grid(axis='y', color='gray', which='both', linewidth=0.2)
sns.despine()
path_out = './' + name + '_' + yname.replace(' ', '_').replace('/', '') + '.png'
plt.savefig(path_out)