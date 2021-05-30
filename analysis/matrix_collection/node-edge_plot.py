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

x = []
y = []
z = []
path = './results/mean/mean_results.csv'
with open(path,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        if row[0] == 'Dataset name':
            continue
        x.append(int(float(row[2])))
        y.append(int(float(row[3])))
        z.append(row[0])

sns.scatterplot(x,y,sizes=10,hue=z, legend=False)

plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

#plt.yticks(np.arange(min(y), max(y)+1, step=100))
plt.xlabel('# nodes')
plt.ylabel('# edges')
plt.title('NODE-EDGE CORRELATION', y=0.98, fontsize=10, fontweight='bold')
#plt.legend(fontsize=8, loc='best', markerscale=2.0)

plt.grid()

path_out = './matrix_collection.png'
plt.savefig(path_out)
