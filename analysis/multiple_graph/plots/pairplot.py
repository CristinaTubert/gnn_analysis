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

name = 'merged'
x = []
y = []
path = '../generic_metrics/results/all/'+ name +'.csv'
'''
with open(path,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        if row[0] == 'Dataset name':
            continue
        x.append(float(row[9]))
        y.append(float(row[8]))
'''
df = pd.read_csv(path)
plt.figure(figsize=(10,8), dpi=80)

pp = sns.pairplot(df, kind='scatter', hue='Dataset name', diag_kind='hist', plot_kws=dict(s=80, edgecolor='white', linewidth=2.5))

path_out = './cluster_density-' + name + '.png'
plt.savefig(path_out)
