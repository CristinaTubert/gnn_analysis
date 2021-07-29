import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
import seaborn as sns
import sys
import pickle
import matplotlib.ticker as ticker

import collections

objects=[]
treated=[]

'''
#maximo = open('degree_list2.txt', 'wb')

with (open('new_degree_list.txt', 'rb')) as openfile:
    while True:
        try:
            obj = pickle.load(openfile)
            print(obj[0])
            # if obj[0][0] == 'Yelp':
            #     print(obj[0])
            #     continue
            
            # else:
            #     pickle.dump(obj, maximo)
        except EOFError:
            break


'''
with (open('new_degree_list.txt', 'rb')) as openfile:
    while True:
        try:
            obj = pickle.load(openfile)
            if obj[0] == 'ogbn-products':
                obj[0] = ['ogbn-products']
            objects.append(obj)

        except EOFError:
            break

degrees = []
names = []
objects.sort(key = lambda x:x[0])
for obj in objects:
    degrees.append(obj[5:])
    names.append(obj[0])
'''
all_degrees=[]
for deg in degrees:
    for d in deg:
        all_degrees.append(int(d[1]))

print(len(all_degrees))
all_names=[]
for i in range(len(names)):
    for d in degrees[i]:
        all_names.append(names[i])

print(len(all_names))
print(min(all_degrees))
'''
data = []
w = []
fig, axes = plt.subplots(5, 2, figsize=(15,27), dpi=300, sharey=True)
colors=['crimson', 'tomato', 'sandybrown', 'gold', 'palegreen', 'mediumseagreen', 'lightskyblue', 'navy', 'mediumpurple', 'palevioletred']

count=0
for i in range(5):
    for j in range(2):
            partial_data = [int(d[1]) for d in degrees[count]]
            data.append(partial_data)
            partial_weight = np.ones(len(partial_data))/len(partial_data)
            w.append(partial_weight)

            axes[i][j].hist(partial_data, bins=100, weights=partial_weight, color= colors[count], label=names[count], alpha=0.8)

            annotate='max degree\n' + str(max(partial_data))
            axes[i][j].annotate(annotate, (max(partial_data), max(partial_weight[partial_data.index(max(partial_data))], 0.00001)), xytext=(max(partial_data)-3, max(partial_weight[partial_data.index(max(partial_data))]+0.001, 0.0001)), xycoords='data', textcoords='data', arrowprops=dict(facecolor='gray', arrowstyle='->'), horizontalalignment='right', verticalalignment='top')
            
            if(j==0):
                axes[i][j].set_ylabel('proportion (log scale)', labelpad=12)

            axes[i][j].yaxis.set_tick_params(labelleft=True)
            axes[i][j].set_xlabel('node degree', labelpad=10)
            count+=1

#for i, (ax,cut) in enumerate(zip(axes.flatten(), names)):
    
#plt.hist(data, weights=w, alpha=1, label=names[2:5])
#print(data)
#print([a*b for a,b in zip(data,w)])
'''
ax = sns.displot(x=all_degrees, hue=all_names, kind='kde', common_norm=False)
ax.set(xscale='symlog', xlim=(-10, None))
'''
plt.suptitle('NODE DEGREE DISTRIBUTION by dataset', weight='bold', fontsize=20)
plt.semilogy()
plt.subplots_adjust(hspace=0.2, top=0.95, bottom=0.07)
fig.legend(fontsize='x-large', ncol=5, loc='lower center')
plt.savefig('d.png')
