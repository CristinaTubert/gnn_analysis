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
path = '../generic_metrics/new_results/all/'
all_files = glob.glob(os.path.join(path, "ogbg-*.csv"))

df_from_each_file = (pd.read_csv(f, sep=',') for f in all_files)
df_merged = pd.concat(df_from_each_file, ignore_index=True)
df_merged.to_csv("../generic_metrics/new_results/all/merged.csv")
    
'''

name = 'merged'
x = []
y = []

# Import Data
path = '../generic_metrics/new_results/all/' + name + '.csv'
df = pd.read_csv(path, low_memory=False)

# Draw Plot
datasets = sorted(list(set(df['Dataset name'])))
plt.figure(figsize=(13,10), dpi= 100)
sns.boxplot(x='Dataset name', y='Average degree', data=df, order=datasets, notch=False)

# Decoration
plt.xticks(rotation=45)
plt.title('Density of graphs by dataset', fontsize=22)

plt.grid(axis='y')
path_out = name + '.png'
plt.savefig(path_out)