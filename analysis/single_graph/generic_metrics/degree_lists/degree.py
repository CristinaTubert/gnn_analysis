import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
import seaborn as sns
import sys
import pickle

import collections

objects=[]
with (open('degree_list.txt', 'rb')) as openfile:
    while True:
        try:
            obj = pickle.load(openfile)
            if obj[3]%10 != 0:
                print(obj[0])
                print(obj[3])
                objects.append(obj)
        except EOFError:
            break

degrees=[]
size_split=[]
for obj in objects:
    degrees.append(obj[5:])
    size_split.append(obj[3])

all_degrees=[]
all_counts=[]
print(type(degrees[0]))
for deg in degrees:
    d,cnt = map(list, zip(*deg))
    all_degrees.append(d)
    all_counts.append(cnt)

Degrees=[item for sublist in all_degrees for item in sublist]
Counts=[item for sublist in all_counts for item in sublist]

Sizes = []
for i in range(len(size_split)):
    for d in all_degrees[i]:
        Sizes.append(size_split[i])

print(len(Degrees))
print(len(Counts))
print(len(Sizes))


sns.displot(x=Degrees, y=Counts, hue=Sizes, colors=['red', 'white', 'black', 'blue'])

plt.savefig('d.png')
