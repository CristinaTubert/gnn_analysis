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

path = '../generic_metrics/results/all/'
all_files = glob.glob(os.path.join(path, "ogbg-*.csv"))

df_from_each_file = (pd.read_csv(f, sep=',') for f in all_files)
df_merged = pd.concat(df_from_each_file, ignore_index=True)
df_merged.to_csv("merged.csv")
'''
name = sys.argv[1]
x = []
y = []
path = '../generic_metrics/results/all/ogbg-' + name + '.csv'
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