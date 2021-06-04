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
pp = sns.pairplot(df, kind='scatter', hue='Dataset name', height=3, x_vars= ['Num nodes', 'Num edges', 'Density'], y_vars=['Cluster coefficient'], plot_kws=dict(s=80, edgecolor='white', linewidth=2.5))
pp.set(xscale='log')

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

path_out = './cluster_density-' + name + '.png'
plt.savefig(path_out)
