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

name = 'merged'
x = []
y = []

# Import Data
path = '../generic_metrics/results/all/' + name + '.csv'
df = pd.read_csv(path, low_memory=False)

# Draw Plot
datasets = sorted(list(set(df['Dataset name'])))
plt.figure(figsize=(13,10), dpi= 80)
sns.boxplot(x='Dataset name', y='Density', data=df, order=datasets, notch=False)

# Decoration
plt.xticks(rotation=45)
plt.title('Box Plot of Density by Dataset', fontsize=22)

plt.grid(axis='y')
path_out = './node-edge/' + name + '.png'
plt.savefig(path_out)