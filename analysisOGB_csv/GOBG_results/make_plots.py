import matplotlib.pyplot as plt
import numpy as np
import csv

x = []
y = []

with open('results_OGBG.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        if row[2] == 'Graph ID':
            continue
        
        x.append(str(row[2]))
        y.append(int(row[5]))

plt.plot(x,y)
plt.xticks(rotation='vertical', fontsize=5)
plt.yticks(np.arange(min(y), max(y)+1, step=100))
plt.xlabel('Graph ID')
plt.ylabel('Num edges')
plt.title('Num edges of graphs in ogbg-code2')
plt.savefig('imatge.png')