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
import matplotlib.patches as mpatches
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

name = 'ogbn-proteins'
yname = 'Average degree'
num_nodes = 132534

# Import Data
path = '../generic_metrics/results/all/' + name + '.csv'

x = []
y1 = []
y2 = []
z = []

with open(path,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        if row[0] == 'Dataset name':
            continue

        if int(row[3])== num_nodes:
            Real_degree=float(row[5])
            continue

        x.append(int(row[3])) #num nodes (split size)
        y1.append(float(row[5])) #real avg degree
        y2.append(float(row[6])) #avg degree

        if row[-1].find('random2') != -1:
            z.append('random')

        else:
            z.append('ordered')


# Draw Plot
fig, axes = plt.subplots(2, 1, figsize=(10,10), dpi=300, sharex=True)
size_split = sorted(list(set(x)))
colors={'ordered':'firebrick', 'random':'lightseagreen'}
box1=sns.boxplot(ax=axes[0], x=x, y=y1, hue=z, order=size_split, notch=False, linewidth=0.5, fliersize=3, palette=colors, meanline=True, whis=1.5, flierprops=dict(markerfacecolor='purple', markeredgecolor='purple'))
box2=sns.boxplot(ax=axes[1], x=x, y=y2, hue=z, order=size_split, notch=False, linewidth=0.5, fliersize=3, palette=colors, meanline=True, whis=1.5, flierprops=dict(markerfacecolor='purple', markeredgecolor='purple'))

axes[0].set_title('WITH edge cuts', fontsize=10, y=1.0, pad=-14, fontstyle='italic')
axes[1].set_title('WITHOUT edge cuts', fontsize=10, y=1.0, pad=-14, fontstyle='italic')

handles, labels = axes[0].get_legend_handles_labels()

for ax in axes:
    ax.legend([],[], frameon=False)
    ax.xaxis.set_tick_params(labelleft=True)
    ax.set_ylabel('average degree', labelpad=10, fontsize=8)
    ax.set_xlabel('# nodes', labelpad=10, fontsize=8)
    plt.setp(ax.lines, color='red')
    ax.tick_params(axis='x',labelsize=8)
    ax.grid(axis='y', color='gray', which='both', linewidth=0.2)
    #ax.spines['top'].set_visible(False)

xz = [(x[i], z[i]) for i in range(0, len(x))]
result = {i:[] for i in xz}
for i,j in zip(xz, y1):
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
xticks = axes[0].get_xticks()


print(sizes)
outlier_i = 0
tick = 0
while outlier_i < len(all_outliers) and tick < len(xticks):
    outlier=all_outliers[outlier_i]

    if (outlier[0][0] == sizes[tick] and outlier[0][1] == 'ordered'):
        axes[0].text(xticks[tick]-0.25, max(y1)+80, '{:.3f}'.format(outlier[1]), horizontalalignment='center', size='xx-small', color='purple', rotation=45)
        outlier_i+=1

    elif (outlier[0][0] == sizes[tick] and outlier[0][1] == 'random'):
        axes[0].text(xticks[tick]+0.25, max(y1)+80, '{:.3f}'.format(outlier[1]), horizontalalignment='center', size='xx-small', color='purple', rotation=45)
        outlier_i+=1

    else:
        tick+=1

axes[0].text(-0.55, max(y1)+90, 'outliers \nratio', horizontalalignment='center', size='xx-small', color='purple')


xz = [(x[i], z[i]) for i in range(0, len(x))]
result = {i:[] for i in xz}
for i,j in zip(xz, y2):
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
xticks = axes[0].get_xticks()


print(sizes)
outlier_i = 0
tick = 0
while outlier_i < len(all_outliers) and tick < len(xticks):
    outlier=all_outliers[outlier_i]

    if (outlier[0][0] == sizes[tick] and outlier[0][1] == 'ordered'):
        axes[1].text(xticks[tick]-0.25, max(y2)+35, '{:.3f}'.format(outlier[1]), horizontalalignment='center', size='xx-small', color='purple', rotation=45)
        outlier_i+=1

    elif (outlier[0][0] == sizes[tick] and outlier[0][1] == 'random'):
        axes[1].text(xticks[tick]+0.25, max(y2)+35, '{:.3f}'.format(outlier[1]), horizontalalignment='center', size='xx-small', color='purple', rotation=45)
        outlier_i+=1

    else:
        tick+=1

axes[1].text(-0.55, max(y2)+40, 'outliers \nratio', horizontalalignment='center', size='xx-small', color='purple')

xticklabels=[]
for s in sizes:
    l = str(s)+ '\n ('+ str('{:.2f}'.format((s/num_nodes) * 100)) + '%)'
    xticklabels.append(l)

plt.xticks(ticks=xticks, labels=xticklabels)

xticks = axes[0].get_xticks()
annotate = 'avg degree\n\n  ' + str('{:.4f}'.format(Real_degree))
axes[0].annotate(annotate, (xticks[-1]+0.5, Real_degree), xytext=(xticks[-1]+0.9, Real_degree), xycoords='data', textcoords='data',  arrowprops=dict(alpha=0.1, arrowstyle='wedge, tail_width=0.5', color='darkcyan'), bbox=dict(boxstyle='round', alpha=0.1, color='darkcyan'), size=10, ha='left', va='center')
# xticks = axes[1].get_xticks()
# axes[1].annotate(annotate, (xticks[-1]+0.5, max(y2)+1), xytext=(xticks[-1]+0.9, max(y2)+1), xycoords='data', textcoords='data',  arrowprops=dict(alpha=0.1, arrowstyle='wedge, tail_width=0.5', color='darkcyan'), bbox=dict(boxstyle='round', alpha=0.1, color='darkcyan'), size=7, ha='left', va='center', annotation_clip=False)

suptitle = yname + ' by split size for ' + name.upper()
plt.suptitle(suptitle, y=0.97, fontsize=11, weight='bold')

plt.subplots_adjust(hspace=0.4, right=0.85)

plt.figlegend(handles, labels, loc='upper right')

path_out = './' + name + '_' + yname.replace(' ', '_').replace('/', '') + '.png'
plt.savefig(path_out)