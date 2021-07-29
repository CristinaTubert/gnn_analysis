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

with (open('community_list.txt', 'rb')) as openfile:
    while True:
        try:
            obj = pickle.load(openfile)
            print(obj[0], obj[3], obj[6])
            # if obj[0] == 'FLickr':
            #     print(obj[0])
            #     continue
            
            # pickle.dump(obj, maximo)
        except EOFError:
            break

'''
treated = []
with (open('community_list.txt', 'rb')) as openfile:
    while True:
        try:
            obj = pickle.load(openfile)
            if not obj[0] in treated:
                objects.append(obj)
                treated.append(obj[0])

        except EOFError:
            break

communities = []
names = []
coverages=[]
modularities=[]
num_comm=[]
objects.sort(key = lambda x:x[0])
for obj in objects:
    communities.append(obj[7:])
    name=obj[0]
    if(name[0] == 'Coauthor'):
        name[0]+='CS'
    names.append(name)
    print(names)
    print(obj[7:])

    coverages.append('{:.2f}'.format(float(obj[5])))
    modularities.append('{:.2f}'.format(float(obj[4])))
    num_comm.append(int(obj[6]))
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
fig, axes = plt.subplots(4, 2, figsize=(15,27), dpi=300, sharey=True)
colors=['crimson', 'tomato', 'sandybrown', 'gold', 'palegreen', 'mediumseagreen', 'lightskyblue']#, 'navy', 'mediumpurple', 'palevioletred']
total_nodes=[7600, 3327, 18333, 2708, 89250, 19717, 169343]
count=0
for i in range(4):
    for j in range(2):
        if count < 7:
            data = communities[count]
            weight = np.ones(len(data))/len(data)
            axes[i][j].hist(data, bins=50, weights=weight, color= colors[count], label=names[count])
            
            if(j==0):
                axes[i][j].set_ylabel('proportion', labelpad=12)
            
            axes[i][j].yaxis.set_tick_params(labelleft=True)
            axes[i][j].set_xlabel('community size', labelpad=10)

            annotate=  str('{:.2f}'.format((max(data)/total_nodes[count])*100)) + '%' + '\nof total nodes'
            axes[i][j].annotate(annotate, (max(data)-max(data)/100, 0.05), xytext=(max(data)-max(data)/100-max(data)/10, 0.15), xycoords='data', textcoords='data', arrowprops=dict(facecolor='gray', arrowstyle='->'), horizontalalignment='center', verticalalignment='top', fontsize=12)
                

            text= str(num_comm[count])+ ' communities\nmodularity score: ' + str(modularities[count]) + '\ncoverage score: ' + str(coverages[count])
            print(num_comm[count]==len(data))
            axes[i][j].annotate(text, xy=(0.72,0.9), fontsize=12, xycoords='axes fraction', horizontalalignment='left', verticalalignment='center', bbox=dict(boxstyle='round', edgecolor='black', facecolor='white', linewidth=0.5))
            count+=1

#for i, (ax,cut) in enumerate(zip(axes.flatten(), names)):
    
#plt.hist(data, weights=w, alpha=1, label=names[2:5])
#print(data)
#print([a*b for a,b in zip(data,w)])
'''
ax = sns.displot(x=all_degrees, hue=all_names, kind='kde', common_norm=False)
ax.set(xscale='symlog', xlim=(-10, None))
'''
plt.suptitle('COMMUNITY SIZE DISTRIBUTION by dataset', weight='bold', fontsize=20)

plt.subplots_adjust(hspace=0.2, top=0.95, bottom=0.07)
fig.legend(fontsize='x-large', ncol=3, loc='lower center')

plt.savefig('c.png')
