import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
import seaborn as sns
import sys
from scipy import stats
from collections import Counter
import math

from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

name = 'merge'
x = []
y = []
path = '../generic_metrics/results/all/'+ name +'.csv'
with open(path,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        if row[0] == 'Dataset name':
            continue
        x.append(float(row[9]))
        y.append(float(row[8]))

df = pd.read_csv(path)
x_vars=['Num nodes', 'Density']
colors={'Actor':'crimson', 'CiteSeer':'tomato', 'CoauthorCS':'sandybrown', 'Cora':'gold', 'Flickr':'palegreen', 'PubMed':'mediumseagreen', 'Yelp':'lightskyblue', 'ogbn-arxiv':'navy', 'ogbn-products':'mediumpurple', 'ogbn-proteins':'palevioletred'}
pp = sns.pairplot(df, kind='scatter', hue='Dataset name', palette=colors, height=5, x_vars= x_vars, y_vars=['Transitivity'], plot_kws=dict(s=80, edgecolor='white', linewidth=0.5, alpha=0.8), )
pp.set(xscale='log')
count=0
for ax in pp.axes.flat:
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.tick_params(axis='y', labelleft=True)
    ax.grid(axis='both', color='gray', linewidth=0.2)
    x_label= x_vars[count]+ ' (log scale)'
    ax.set_xlabel(x_label)
    count+=1

pp._legend.remove()
plt.tight_layout(pad=2.0)
plt.subplots_adjust(bottom=0.35, wspace=0.2, top=0.85)

'''
plt.scatter(x,y,s=5)
plt.plot(x,y)

plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

#plt.semilogy()

#plt.yticks(np.arange(min(y), max(y)+1, step=100))
plt.xlabel('density')
plt.ylabel('cluster coefficient')
plt.title('cluster coefficient - density CORRELATION', fontsize=10, fontweight='bold')
#plt.legend(fontsize=8, loc='best', markerscale=2.0)

plt.grid()
'''
plt.legend(loc='upper center', fontsize='small', ncol=10, bbox_to_anchor=(-0.16,-0.4), shadow=True)
path_out = './cluster_density-' + name + '.png'
plt.savefig(path_out, dpi=200)
