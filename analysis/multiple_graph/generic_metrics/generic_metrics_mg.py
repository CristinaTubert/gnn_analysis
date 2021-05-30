import time
import pandas as pd
import numpy as np
import networkx as nx
import networkx.algorithms.community as nxcom
import random as rand
import csv
import os.path

import collections
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

import ogb.graphproppred as ogbg
from torch_geometric.data import Data
import torch_geometric.utils as utils
import torch_geometric
import torch

METRICS = -1
DEGREE_HIST = -1
COM_HIST = -1
NUM_GRAPHS = -1
TYPE_SPLIT = ''
METRICS_RESULTS = ''

values_dict = {}

def choose_dataset():
    name = input('Choose dataset graph prediction [molhiv, molpcba, ppa, code]: ')
    name = 'ogbg-' + name
    dataset = ogbg.GraphPropPredDataset(name=name, root='/home/ctubert/tfg/gitprojects/gnn_analysis/analysis/datasets')
    fsplit = input('Choose dataset first split [train, valid, test, no-split]: ')

    ini_dict(name, fsplit)

    return (dataset, fsplit)

def choose_globals(num_graphs):
  global NUM_GRAPHS
  text = 'Choose NUM_GRAPHS [0, ' + str(num_graphs) + ']: '
  NUM_GRAPHS = int(input(text))

  if METRICS:
    global METRICS_RESULTS
    METRICS_RESULTS = input('Choose METRICS_RESULTS [mean, all]: ')

  global TYPE_SPLIT
  TYPE_SPLIT = input('Choose TYPE_SPLIT [random, ordered]: ')

def first_split(dataset, fsplit):
    if fsplit == 'no-split':
        graphs_ini = list(range(0, len(dataset)))
    else:
        split_idx = dataset.get_idx_split()
        graphs_ini = split_idx[fsplit]

    return graphs_ini

def second_split(graphs_ini, i):
  if TYPE_SPLIT == 'random':
    rand.shuffle(graphs_ini)
    graphs = graphs_ini[0:NUM_GRAPHS]

  elif TYPE_SPLIT == 'ordered':
    graphs = graphs_ini[i:(i+NUM_GRAPHS)]

  return graphs

def get_nodes_edges(G):
    nodes = list(range(0, G[0]['num_nodes']))
    edges = G[0]['edge_index']

    return nodes, edges

def OGB_to_nx(nodes, edges):
  edges_tensor = torch.LongTensor([x for x in edges])
  try:
    undirected = utils.is_undirected(edges_tensor)
  except:
    undirected = True
    print('UNDIRECTED ERROR')

  edge_list = []
  for i in range(len(edges[0])):
    edge_list.append((int(edges[0][i]), int(edges[1][i])))

  if undirected:
    G = nx.Graph()
  else:
    G = nx.DiGraph()
    
  G.add_nodes_from(nodes)
  G.add_edges_from(edge_list)

  return (G, undirected)

def graph_processing(G, undirected):
  num_nodes = G.number_of_nodes()
  num_edges = G.number_of_edges()

  degree_sequence = sorted([d for n, d in G.degree()])
  avg_degree = sum(degree_sequence) / num_nodes

  avg_clustering = nx.algorithms.cluster.average_clustering(G)

  density = nx.density(G)
 
  values_dict['Directed'].append(not undirected)
  values_dict['Num nodes'].append(num_nodes)
  values_dict['Num edges'].append(num_edges)
  values_dict['Average degree'].append(avg_degree)
  values_dict['Average clustering'].append(avg_clustering)
  values_dict['Density'].append(density)

def get_biggest_CC(G, undirected):
  if undirected:
    CC = [G.subgraph(c) for c in sorted(nx.algorithms.components.connected_components(G), key=len, reverse=True)]
  else:
    CC = [G.subgraph(c) for c in sorted(nx.algorithms.components.strongly_connected_components(G), key=len, reverse=True)]

  num_CC = len(CC)
  cc = CC[0]

  values_dict['Num CC'].append(num_CC)

  return cc

def CC_processing(cc, undirected):
  num_nodes = cc.number_of_nodes()
  num_edges = cc.number_of_edges()

  avg_path_length = nx.average_shortest_path_length(cc)
  diameter = nx.diameter(cc)
  radius = nx.radius(cc)
  node_connectivity = nx.algorithms.connectivity.connectivity.node_connectivity(cc)
  try:
    edge_connectivity = nx.algorithms.connectivity.connectivity.edge_connectivity(cc)
  except:
    edge_connectivity = -1
    print('EDGE CONNECTIVITY ERROR')
  finally:
    values_dict['BCC num nodes'].append(num_nodes)
    values_dict['BCC num edges'].append(num_edges)
    values_dict['BCC average path length'].append(avg_path_length)
    values_dict['BCC diameter'].append(diameter)
    values_dict['BCC radius'].append(radius)
    values_dict['BCC node connectivity'].append(node_connectivity)
    values_dict['BCC edge connectivity'].append(edge_connectivity)

def community_detection(G, undirected):
  try:
    gm_lcom = list(nxcom.greedy_modularity_communities(G))
    gm_communities = len(gm_lcom)
    gm_modularity = nxcom.modularity(G, gm_lcom)
    gm_coverage = nxcom.coverage(G, gm_lcom)

    # result = nxcom.girvan_newman(G)
    # print(list(result))
    # gn_lcom = next(result)
    # print(gn_lcom)
    # gn_communities = len(gn_lcom)
    # gn_modularity = nxcom.modularity(G, gn_lcom)
    # gn_coverage = nxcom.coverage(G, gn_lcom)

  except:
    if not undirected:
      try:
        G = nx.to_undirected(G)
        values_dict['Graph ID'][-1] += '*'

        gm_lcom = list(nxcom.greedy_modularity_communities(G))
        gm_communities = len(gm_lcom)
        gm_modularity = nxcom.modularity(G, gm_lcom)
        gm_coverage = nxcom.coverage(G, gm_lcom)

        # result = nxcom.girvan_newman(G)
        # gn_lcom = next(result)
        # gn_communities = len(gn_lcom)
        # gn_modularity = nxcom.modularity(G, gn_lcom)
        # gn_coverage = nxcom.coverage(G, gn_lcom)

      except:
        values_dict['Graph ID'][-1] += '*'
        gm_communities = 0
        gm_modularity = 0
        gm_coverage = 0
        print('COMMUNITY ERROR')

    else:
      values_dict['Graph ID'][-1] += '*'
      gm_communities = 0
      gm_modularity = 0
      gm_coverage = 0
      print('COMMUNITY ERROR')

  finally:
    values_dict['BCC num greedy modularity communities'].append(gm_communities)
    values_dict['Modularity greedy modularity communities'].append(gm_modularity)
    values_dict['Coverage greedy modularity communities'].append(gm_coverage)
    # values_dict['BCC girvan newman communities'].append(gn_communities)
    # values_dict['Modularity girvan newman communities'].append(gn_modularity)
    # values_dict['Coverage girvan newman communities'].append(gn_coverage)

    if COM_HIST and gm_communities > 0:
      histogram_community(gm_lcom)
      # histogram_community(gn_lcom, 'gn')

def histogram_community(communities):
  print('Generating communities plots...')
  com_sizes = []
  for c in range(0, len(communities)):
    com_sizes.append(len(communities[c]))

  sizeCount = collections.Counter(com_sizes)
  size, cnt = zip(*sizeCount.items())

  fig, ax = plt.subplots()
  plt.bar(size, cnt, color="b", align='edge')
  plt.xticks(rotation='vertical', fontsize=8)

  plt.title("Greedy modularity communities size distribution")
  plt.ylabel("Count")
  plt.xlabel("Size")

  if max(size) < 20:
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_major_formatter('{x:.0f}')

  elif max(size) < 175:
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_major_formatter('{x:.0f}')
    ax.xaxis.set_minor_locator(MultipleLocator(1))

  else:
    ax.xaxis.set_major_locator(MultipleLocator(5*round((max(size)//7)/5)))
    ax.xaxis.set_major_formatter('{x:.0f}')
    ax.xaxis.set_minor_locator(MultipleLocator(5*round((max(size)//35)/5)))
  
  plt.xlim([0, max(size)+1])

  if max(cnt) < 20:
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_formatter('{x:.0f}')

  elif max(cnt) < 175:
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_formatter('{x:.0f}')
    ax.yaxis.set_minor_locator(MultipleLocator(1))

  else:
    ax.yaxis.set_major_locator(MultipleLocator(5*round((max(cnt)//7)/5)))
    ax.yaxis.set_major_formatter('{x:.0f}')
    ax.yaxis.set_minor_locator(MultipleLocator(5*round((max(cnt)//35)/5)))

  plt.ylim([0, max(cnt)+1])
    
  name = values_dict['Dataset name']
  graph_id = values_dict['Graph ID'][-1]
  if graph_id[-1] == '*':
    graph_id = graph_id[:-1]

  if TYPE_SPLIT == 'random':
    path = "./community_hist/" + name + "/random/" "/com" + str(NUM_GRAPHS) + "_" + str(graph_id) + ".png" 
  
  elif TYPE_SPLIT == 'ordered':
    path = "./community_hist/" + name + "/com" + str(NUM_GRAPHS) + "_" + str(graph_id) + ".png"

  plt.savefig(path)
  plt.clf()

def histogram_degree(G, undirected):
  degree_sequence = sorted([d for n, d in G.degree()])  # degree sequence
  #take 98%
  #print(degree_sequence)
  k = int(len(degree_sequence)*1)

  degreeCount = collections.Counter(degree_sequence[:k])
  deg, cnt = zip(*degreeCount.items())

  fig, ax = plt.subplots()
  plt.bar(deg, cnt, color="b", align='edge')
  plt.xticks(rotation='vertical', fontsize=8)

  plt.title("Degree Histogram")
  plt.ylabel("Count")
  plt.xlabel("Degree")

  '''
  plt.xticks([d + 0.4 for d in deg])
  ax.set_xticklabels([d for d in deg])
  '''
  if max(deg) < 20:
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_major_formatter('{x:.0f}')

  elif max(deg) < 175:
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_major_formatter('{x:.0f}')
    ax.xaxis.set_minor_locator(MultipleLocator(1))

  else:
    ax.xaxis.set_major_locator(MultipleLocator(5*round((max(deg)//7)/5)))
    ax.xaxis.set_major_formatter('{x:.0f}')
    ax.xaxis.set_minor_locator(MultipleLocator(5*round((max(deg)//35)/5)))
  
  plt.xlim([0, max(deg)+1])

  if max(cnt) < 20:
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_formatter('{x:.0f}')

  elif max(cnt) < 175:
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_formatter('{x:.0f}')
    ax.yaxis.set_minor_locator(MultipleLocator(1))

  else:
    ax.yaxis.set_major_locator(MultipleLocator(5*round((max(cnt)//7)/5)))
    ax.yaxis.set_major_formatter('{x:.0f}')
    ax.yaxis.set_minor_locator(MultipleLocator(5*round((max(cnt)//35)/5)))

  plt.ylim([0, max(cnt)+1])
    
  text = 'max degree = ' + str(max(degree_sequence))
  plt.annotate(text, xy=(0.70, 0.90), xycoords='axes fraction')
  
  '''# draw graph in inset
  plt.axes([0.4, 0.4, 0.5, 0.5])
  Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
  pos = nx.spring_layout(G)
  plt.axis("off")
  nx.draw_networkx_nodes(G, pos, node_size=20)
  nx.draw_networkx_edges(G, pos, alpha=0.4)'''

  name = values_dict['Dataset name']
  graph_id = values_dict['Graph ID'][-1]
  if graph_id[-1] == '*':
    graph_id = graph_id[:-1]

  if TYPE_SPLIT == 'random':
    path = "./degree_hist/" + name + "/random/" "/deg" + str(NUM_GRAPHS) + "_" + str(graph_id) + ".png" 
  
  elif TYPE_SPLIT == 'ordered':
    path = "./degree_hist/" + name + "/deg" + str(NUM_GRAPHS) + "_" + str(graph_id) + ".png"

  plt.savefig(path)
  plt.clf()

def ini_dict(name, fsplit):
  values_dict['Dataset name'] = name
  values_dict['First split'] = fsplit
  values_dict['Num graphs'] = -1
  values_dict['Graph ID'] = []
  values_dict['Directed'] = []
  values_dict['Num nodes'] = []
  values_dict['Num edges'] = []
  values_dict['Average degree'] = []
  values_dict['Average clustering'] = []
  values_dict['Density'] = []
  values_dict['Num CC'] = []
  values_dict['BCC num nodes'] = []
  values_dict['BCC num edges'] = []
  values_dict['BCC average path length'] = []
  values_dict['BCC diameter'] = []
  values_dict['BCC radius'] = []
  values_dict['BCC node connectivity'] = []
  values_dict['BCC edge connectivity'] = []
  values_dict['BCC num greedy modularity communities'] = []
  values_dict['Modularity greedy modularity communities'] = []
  values_dict['Coverage greedy modularity communities'] = []
  values_dict['Execution time'] = []
  values_dict['Type split'] = ''

def mean_dict():
  for key,value in values_dict.items():
    if type(value) == list:
      values_dict[key] = sum(value)/len(value)

def get_subdict(i):
  subdict = {}
  for key,value in values_dict.items():
    if type(value) == list:
      value = value[i]
    subdict[key] = value
  return subdict

def write_csv_all():
  name = values_dict['Dataset name']
  file_name = name + '.csv'
  path = 'results/all/' + file_name

  if os.path.exists(path):
    with open(path, 'a', newline='') as f:
      w = csv.DictWriter(f, values_dict.keys())
      lim = len(values_dict['Execution time'])
      for i in range(lim):
        subdict = get_subdict(i)
        w.writerow(subdict)
  else:
    with open(path, 'w', newline='') as f:
      w = csv.DictWriter(f, values_dict.keys())
      w.writeheader()
      lim = len(values_dict['Execution time'])
      for i in range(lim):
        subdict = get_subdict(i)
        w.writerow(subdict)

def write_csv_mean():
  path = 'results/mean/mean_results.csv'

  if os.path.exists(path):
    with open(path, 'a', newline='') as f:
      w = csv.DictWriter(f, values_dict.keys())
      w.writerow(values_dict)

  else:
    with open(path, 'w', newline='') as f:
      w = csv.DictWriter(f, values_dict.keys())
      w.writeheader()
      w.writerow(values_dict)

def graph_to_characterization(G, undirected):
  time_ini = time.time()

  #('Graph processing...')
  graph_processing(G, undirected)
  cc = get_biggest_CC(G, undirected)
  #print('CC processing...')
  CC_processing(cc, undirected)
  #print('Communities processing...')
  community_detection(cc, undirected)

  time_end = time.time() - time_ini

  values_dict['Execution time'].append(time_end)

def split_to_results(G, undirected):
  if METRICS: 
    graph_to_characterization(G, undirected)

  if DEGREE_HIST:
    print('Generating degree plots...')
    histogram_degree(G, undirected)

  if (not METRICS) and COM_HIST:
    cc = get_biggest_CC(G, undirected)
    community_detection(cc, undirected)

def split_control(dataset, fsplit):
  graphs_ini = first_split(dataset, fsplit)
  values_dict['Num graphs'] = len(graphs_ini)
  choose_globals(len(graphs_ini))

  if TYPE_SPLIT == 'ordered':
    final_split = len(graphs_ini) - NUM_GRAPHS
    ini = int(input('Choose ini_split [0, ' + str(final_split) + ']: '))
    
    values_dict['Type split'] = 'ordered graphs since ' + str(ini)
    
  elif TYPE_SPLIT == 'random':
    ini = -1
    values_dict['Type split'] = 'random graphs'

  graphs = second_split(graphs_ini, ini)

  for i in graphs:
    print('Iteration ' + str(i))
    values_dict['Graph ID'].append(str(i))
    G = dataset[i]
    nodes, edges = get_nodes_edges(G)
    G, undirected = OGB_to_nx(nodes, edges)
    split_to_results(G, undirected)

  if METRICS:
    print('Writing metrics...')
    if METRICS_RESULTS == 'mean':
      mean_dict()
      write_csv_mean()
      
    elif METRICS_RESULTS == 'all':
      write_csv_all()

def main():
  dataset, fsplit = choose_dataset()
  global METRICS 
  METRICS = bool(int(input('Do you want to perform generic metric characterization? [yes=1/ no=0]: ')))
  global DEGREE_HIST 
  DEGREE_HIST = bool(int(input('Do you want to generate degree distribution histograms? [yes=1/ no=0]: ')))
  global COM_HIST 
  COM_HIST = bool(int(input('Do you want to generate community distribution histograms? [yes=1/ no=0]: ')))
  split_control(dataset, fsplit)

def test():
    name = input('Choose dataset node prediction [molhiv, molpcba, ppa, code]: ')
    name = 'ogbg-' + name
    dataset = ogbg.GraphPropPredDataset(name=name, root='/home/ctubert/tfg/gitprojects/gnn_analysis/analysis/datasets')
    print(len(dataset))
    print(type(dataset))
    print(type(dataset[0]))
    for d in dataset[455]:
      print(type(d))
      print(d)
      print('\n')

if __name__ == '__main__':
  test()