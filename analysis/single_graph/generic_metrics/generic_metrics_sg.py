import time
import pandas as pd
import numpy as np
import networkx as nx
import networkx.algorithms.community as nxcom
import random as rand
import csv
import os.path
import pickle


import collections
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

#import cugraph as cnx

import ogb.nodeproppred as ogbn
from torch_geometric.data import Data
import torch_geometric.utils as utils
import torch_geometric
import torch

from torch_geometric.datasets import Planetoid
import torch_geometric.datasets as tg
from torch_geometric.datasets import SNAPDataset

METRICS = -1
DEGREE_HIST = -1
COM_HIST = -1
SIZE_SPLIT = -1
TYPE_SPLIT = ''
METRICS_RESULTS = ''

values_dict = {}

def choose_dataset():
  data = input('Choose data [OGB, Plan]: ')

  if data == 'OGB':
    name = input('Choose OGB dataset node prediction [arxiv, products, proteins, mag, papers100M]: ')
    name = 'ogbn-' + name
    dataset = ogbn.NodePropPredDataset(name=name, root='/home/ctubert/tfg/gitprojects/gnn_analysis/analysis/datasets')
    fsplit = input('Choose dataset first split [train, valid, test, no-split]: ')

  elif data == 'Plan':
    name = input('Choose Planetoid dataset [Cora, CiteSeer, PubMed]: ')
    
    name='Yelp'
    root='/home/ctubert/tfg/gitprojects/gnn_analysis/analysis/datasets/Yelp'
    dataset = tg.Yelp(root=root)
    print(dataset)
    print(dataset[0])
    
    #dataset = Planetoid(name=name, root='/home/ctubert/tfg/gitprojects/gnn_analysis/analysis/datasets')
    fsplit = 'no-split'

  ini_dict(fsplit)
  
  return (name, data, dataset, fsplit)

def choose_globals(num_nodes):
  global SIZE_SPLIT 
  text = 'Choose SIZE_SPLIT [0, ' + str(num_nodes) + ']: '
  SIZE_SPLIT = int(input(text))

  if METRICS:
    global METRICS_RESULTS
    METRICS_RESULTS = input('Choose METRICS_RESULTS [mean, all]: ')

  global TYPE_SPLIT
  TYPE_SPLIT = input('Choose TYPE_SPLIT [random, ordered, random2]: ')

def first_split(data, dataset, fsplit):
  if data == 'OGB':
    if fsplit == 'no-split':
      nodes_ini = list(range(0, dataset[0][0]['num_nodes']))
    else:
      split_idx = dataset.get_idx_split()
      nodes_ini = split_idx[fsplit]

    edges_ini = dataset[0][0]['edge_index']
    # maxi = dataset[0][0]['num_nodes']
    # print(0 in edges_ini[0])
    # print(0 in edges_ini[1])
    # print(maxi in edges_ini[0])
    # print(maxi in edges_ini[1])

  elif data == 'Plan':
    nodes_ini = list(range(0, len(dataset[0].y))) #always no split
    print(dataset)
    print(dataset[0])
    print(len(dataset[0].y))
    edges_ini = -1 #no matter

  return (nodes_ini, edges_ini)

def second_split(G_all, nodes_ini, i):
  if TYPE_SPLIT == 'random':
    rand.shuffle(nodes_ini)
    nodes = nodes_ini[0:SIZE_SPLIT]

  elif TYPE_SPLIT == 'ordered' or TYPE_SPLIT == 'random2':
    j = min( (i+SIZE_SPLIT), len(nodes_ini) )
    nodes = nodes_ini[i:j]

  G = G_all.subgraph(nodes)
  print(type(G))
  return G, nodes

def OGB_to_nx(fsplit, nodes_ini, edges_ini):
  edges_tensor = torch.LongTensor([x for x in edges_ini])
  undirected = utils.is_undirected(edges_tensor)

  if (fsplit != 'no-split'):
    nodes_tensor = torch.LongTensor([x for x in nodes_ini])
    edges_ini, _ = utils.subgraph(nodes_tensor, edges_tensor)

  edge_list = []
  for i in range(len(edges_ini[0])):
    edge_list.append((int(edges_ini[0][i]), int(edges_ini[1][i])))

  if undirected:
    G = nx.Graph()
  else:
    G = nx.DiGraph()
    
  G.add_nodes_from(nodes_ini)
  G.add_edges_from(edge_list)

  return (G, undirected)

def planetoid_to_nx(dataset):
  D = Data(dataset[0].x, dataset[0].edge_index, dataset[0].y)
  #print(dataset[0].edge_index)
  #G = utils.to_networkx(D)
  G = utils.to_networkx(D, to_undirected=True)

  undirected = not nx.is_directed(G)

  return G, undirected

def graph_processing(G, G_all, nodes, undirected, i):
  num_nodes = G.number_of_nodes()
  num_edges = G.number_of_edges()

  name = values_dict['Dataset name'][-1]
  if name[-1] == '*':
    name = name[:-1]
  
  path_picke = path = "./degree_lists/new_degree_list.txt" 
  degrees = [values_dict['Dataset name'], values_dict['First split'], TYPE_SPLIT, SIZE_SPLIT, i]
  deg = G_all.degree(nodes)
  degrees+= deg

  # with open(path_picke, 'ab') as fp:
  #   pickle.dump(degrees, fp)

  real_degree_sequence = sorted([d for n, d in deg])
  real_avg_degree = sum(real_degree_sequence) / num_nodes

  degree_sequence = sorted([d for n, d in G.degree()])
  avg_degree = sum(degree_sequence) / num_nodes

  edge_cut = 0

  for n in nodes:
    rd = G_all.degree(n)
    d = G.degree(n)
    edge_cut += rd - d
  '''
  avg_clustering = nx.algorithms.cluster.average_clustering(G)
  global_clustering = nx.algorithms.cluster.transitivity(G)

  density = nx.density(G)
  '''
  avg_clustering = 0
  global_clustering = 0

  density = 0

  values_dict['Num nodes'].append(num_nodes)
  values_dict['Num edges'].append(num_edges)
  values_dict['Real average degree'].append(real_avg_degree)
  values_dict['Average degree'].append(avg_degree)
  values_dict['Edge cut'].append(edge_cut)
  values_dict['Average clustering'].append(avg_clustering)
  values_dict['Transitivity'].append(global_clustering)
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

  # try:
  #   avg_path_length = nx.average_shortest_path_length(cc)
  #   diameter = nx.diameter(cc)
  #   radius = nx.radius(cc)
  #   node_connectivity = nx.algorithms.connectivity.connectivity.node_connectivity(cc)
  #   edge_connectivity = nx.algorithms.connectivity.connectivity.edge_connectivity(cc)

  # except:
  
  avg_path_length = None
  diameter = None
  radius = None
  node_connectivity = None
  edge_connectivity = None

  values_dict['BCC num nodes'].append(num_nodes)
  values_dict['BCC num edges'].append(num_edges)
  values_dict['BCC average path length'].append(avg_path_length)
  values_dict['BCC diameter'].append(diameter)
  values_dict['BCC radius'].append(radius)
  values_dict['BCC node connectivity'].append(node_connectivity)
  values_dict['BCC edge connectivity'].append(edge_connectivity)

def community_detection(G, undirected, i):

  try:
    '''
    gm_lcom = list(nxcom.greedy_modularity_communities(G))
    gm_communities = len(gm_lcom)
    gm_modularity = nxcom.modularity(G, gm_lcom)
    gm_coverage = nxcom.coverage(G, gm_lcom)
    '''

    gm_lcom = None
    gm_communities = None
    gm_modularity = None
    gm_coverage = None

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
        values_dict['Dataset name'][-1] += '*'

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
        values_dict['Dataset name'][-1] += '*'
        gm_communities = 0
        gm_modularity = 0
        gm_coverage = 0
        print('COMMUNITY ERROR')

    else:
      values_dict['Dataset name'][-1] += '*'
      gm_communities = 0
      gm_modularity = 0
      gm_coverage = 0
      print('COMMUNITY ERROR')

  finally:
    '''
    try:
      path_picke = path = "./communities_lists/community_list.txt" 
      communities = [values_dict['Dataset name'], values_dict['First split'], TYPE_SPLIT, SIZE_SPLIT, gm_modularity, gm_coverage, gm_communities]

      set_com = []
      for com in gm_lcom:
        set_com.append(len(com))

      communities+= set_com
      with open(path_picke, 'ab') as fp:
        pickle.dump(communities, fp)
    except:
      print('')

    '''
    values_dict['BCC num greedy modularity communities'].append(gm_communities)
    values_dict['Modularity greedy modularity communities'].append(gm_modularity)
    values_dict['Coverage greedy modularity communities'].append(gm_coverage)
    # values_dict['BCC girvan newman communities'].append(gn_communities)
    # values_dict['Modularity girvan newman communities'].append(gn_modularity)
    # values_dict['Coverage girvan newman communities'].append(gn_coverage)

    # try:
    #   print('Cliques')

    #   if nx.is_directed(G):
    #     G_un=nx.to_undirected(G)
    #   else:
    #     G_un=G

    #   max_clique = len(max( list(nx.find_cliques(G_un)), key=lambda x:len(x) ))
    #   values_dict['Max clique'].append(max_clique)
    
    # except:
    #   print('ERROR CLIQUE')
    #   values_dict['Max clique'].append(-1)


    values_dict['Max clique'].append(0)

    if COM_HIST and gm_communities > 0:
      histogram_community(gm_lcom, i)
      # histogram_community(gn_lcom, 'gn')

def histogram_community(communities, i):
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
    
  name = values_dict['Dataset name'][-1]
  if name[-1] == '*':
    name = name[:-1]

  if TYPE_SPLIT == 'random':
    path = "./community_hist/" + name + "/random/" "/com" + str(SIZE_SPLIT) + "_" + str(i) + ".png" 
  
  elif TYPE_SPLIT == 'ordered' or TYPE_SPLIT == 'random2':
    path = "./community_hist/" + name + "/com" + str(SIZE_SPLIT) + "_" + str(i) + ".png"

  plt.savefig(path)
  plt.clf()

def histogram_degree(G, G_all, nodes, undirected, i):
  degree_sequence = sorted([d for n, d in G_all.degree(nodes)])  # degree sequence
  #take 98%
  #print(degree_sequence)
  k = int(len(degree_sequence)*0.98)

  degreeCount = collections.Counter(degree_sequence[:k])
  deg, cnt = zip(*degreeCount.items())

  fig, ax = plt.subplots()
  plt.bar(deg, cnt, color="b", align='edge')
  plt.xticks(rotation='vertical', fontsize=8)

  plt.title("Degree Histogram (0.98)")
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

  name = values_dict['Dataset name'][-1]
  if name[-1] == '*':
    name = name[:-1]

  if TYPE_SPLIT == 'random':
    path = "./degree_hist/" + name + "/random/" "/deg" + str(SIZE_SPLIT) + "_" + str(i) + ".png" 
  
  elif TYPE_SPLIT == 'ordered' or TYPE_SPLIT == 'random2':
    path = "./degree_hist/" + name + "/deg" + str(SIZE_SPLIT) + "_" + str(i) + ".png"

  plt.savefig(path)
  plt.clf()

def ini_dict(fsplit):
  values_dict['Dataset name'] = []
  values_dict['Directed'] = -1
  values_dict['First split'] = fsplit
  values_dict['Num nodes'] = []
  values_dict['Num edges'] = []
  values_dict['Real average degree'] = []
  values_dict['Average degree'] = []
  values_dict['Edge cut'] = []
  values_dict['Average clustering'] = []
  values_dict['Density'] = []
  values_dict['Transitivity'] = []
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
  values_dict['Max clique'] = []
  # values_dict['BCC girvan newman communities'] = []
  # values_dict['Modularity girvan newman communities'] = []
  # values_dict['Coverage girvan newman communities'] = []
  values_dict['Execution time'] = []
  values_dict['Type split'] = ''

def mean_dict():
  name = values_dict['Dataset name'][0]
  for n in values_dict['Dataset name']:
    if n[-1] == '*':
      name = n
    break
  values_dict['Dataset name'] = name
  for key,value in values_dict.items():
    if type(value) == list:
      values_dict[key] = sum(value)/len(value)

def get_subdict(i):
  subdict = {}
  for key,value in values_dict.items():
    if type(value) == list:
      print(key)
      value = value[i]
    subdict[key] = value
  return subdict

def write_csv_all():
  name = values_dict['Dataset name'][-1]
  if name[-1] == '*':
    name = name[:-1]

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

def graph_to_characterization(G, G_all, nodes, undirected, i):
  values_dict['Directed'] = (not undirected)

  time_ini = time.time()

  print('Graph processing...')
  graph_processing(G, G_all, nodes, undirected, i)
  cc = get_biggest_CC(G, undirected)
  print('CC processing...')
  CC_processing(cc, undirected)
  print('Communities processing...')
  community_detection(cc, undirected, i)

  time_end = time.time() - time_ini

  values_dict['Execution time'].append(time_end)

def split_to_results(name, G_all, nodes_ini, undirected, i):
  values_dict['Dataset name'].append(name)
  G, nodes = second_split(G_all, nodes_ini, i)
  print('Graph generated')

  if METRICS: 
    graph_to_characterization(G, G_all, nodes, undirected, i)

  if DEGREE_HIST:
    print('Generating degree plots...')
    histogram_degree(G, G_all, nodes, undirected, i)

  if (not METRICS) and COM_HIST:
    cc = get_biggest_CC(G, undirected)
    community_detection(cc, undirected, i)

def split_control(name, data, dataset, fsplit):
  nodes_ini, edges_ini = first_split(data, dataset, fsplit)
  choose_globals(len(nodes_ini))

  if TYPE_SPLIT == 'random': 
    lim = int(input('Choose number of iterations for random splits: '))
    step = 1
    values_dict['Type split'] = str(lim) + ' random iterations'

  elif TYPE_SPLIT == 'ordered' or TYPE_SPLIT == 'random2':
    lim = len(nodes_ini)
    print(lim)
    total_it = lim//SIZE_SPLIT 
    print(total_it)
    num_it = int(input('Choose number of iterations for ordered/random2 splits [1, ' + str(total_it) + ']: '))
    print(num_it)
    use_nodes = num_it * SIZE_SPLIT
    gap_nodes = lim - use_nodes
    print(gap_nodes)
    if num_it == 1:
      num_gap = gap_nodes//(num_it)
    else:
      num_gap = gap_nodes//(num_it-1)
      print(num_gap)
    step = num_gap+SIZE_SPLIT
    if num_it == 1:
      res = gap_nodes%(num_it)
    else:  
      res = gap_nodes%(num_it-1)
    print(step)
    print(res)
    
    if TYPE_SPLIT == 'ordered':
      values_dict['Type split'] = str(num_it) + ' ordered iterations'
    elif TYPE_SPLIT == 'random2':
      values_dict['Type split'] = str(num_it) + ' random2 iterations'

  print('Generating great graph...')
  if data == 'OGB':
    G_all, undirected = OGB_to_nx(fsplit, nodes_ini, edges_ini)
  elif data == 'Plan':
    G_all, undirected = planetoid_to_nx(dataset)
  
  if TYPE_SPLIT == 'random2':
    rand.shuffle(nodes_ini)

  i = 0
  while i < lim:
    print('Iteration ' + str(i))
    split_to_results(name, G_all, nodes_ini, undirected, i)
    i+=step
    if (TYPE_SPLIT == 'ordered'  or TYPE_SPLIT == 'random2') and res > 0:
      i+=1
      res-=1
  
  if METRICS:
    print('Writing metrics...')
    if METRICS_RESULTS == 'mean':
      mean_dict()
      write_csv_mean()
      
    elif METRICS_RESULTS == 'all':
      write_csv_all()

def main():
  name, data, dataset, fsplit = choose_dataset()
  global METRICS 
  METRICS = bool(int(input('Do you want to perform generic metric characterization? [yes=1/ no=0]: ')))
  global DEGREE_HIST 
  DEGREE_HIST = bool(int(input('Do you want to generate degree distribution histograms? [yes=1/ no=0]: ')))
  global COM_HIST 
  COM_HIST = bool(int(input('Do you want to generate community distribution histograms? [yes=1/ no=0]: ')))
  split_control(name, data, dataset, fsplit)



# def second_split(nodes_ini, edges_ini, i):
#   if TYPE_SPLIT == 'random':
#     rand.shuffle(nodes_ini)
#     nodes = nodes_ini[0:SIZE_SPLIT]

#   elif TYPE_SPLIT == 'ordered':
#     j = min( (i+SIZE_SPLIT), len(nodes_ini) )
#     nodes = nodes_ini[i:j]

#   nodes_tensor = torch.LongTensor([x for x in nodes])
#   edges_tensor = torch.LongTensor([x for x in edges_ini])

#   edges, _ = utils.subgraph(nodes_tensor, edges_tensor, num_nodes=len(nodes_ini))

#   return (nodes, edges)

def test():

  # l = [1,2,3,4,5,6,7,8]
  # edges= [[1, 6,9,8],[2,10,11,7]]


  # edge_index_tensor = torch.LongTensor([x for x in edges])
  # nodes_subset_tensor = torch.LongTensor([x for x in l])
  # edge_index, _ = utils.subgraph(nodes_subset_tensor, edge_index_tensor)

  # edge_list = []
  # for k in range(len(edge_index[0])):
  #   edge_list.append((int(edge_index[0][k]), int(edge_index[1][k])))


  # G = nx.Graph()
  # G.add_nodes_from(l)
  # G.add_edges_from(edge_list)
  # print(G.number_of_edges())
  # print(G.edges)
  # print(G.nodes)
  # sub = [11, 9, 6, 3]
  # G2 = G.subgraph(sub)
  # print(G2)
  # print(G2.edges)
  # print(G2.nodes)

  # name = input('Choose OGB dataset node prediction [arxiv, products, proteins, mag, papers100M]: ')
  # name = 'ogbn-' + name
  # dataset = ogbn.NodePropPredDataset(name=name, root='/home/ctubert/tfg/gitprojects/gnn_analysis/analysis/datasets')
  # print(type(dataset))
  # print(type(dataset[0]))
  # print(dataset[0])
  # for d in dataset[0]:
  #   print(type(d))
  #   print(d)
  #   print('\n')

  dataset = Planetoid(name='Cora', root='/home/ctubert/tfg/gitprojects/gnn_analysis/analysis/datasets')
  print(utils.homophily_ratio(dataset[0].edge_index, dataset[0].y))
  D = Data(dataset[0].x, dataset[0].edge_index, dataset[0].y)
  print(dataset[0])
  G = utils.to_networkx(D)
  print(G.number_of_nodes())
  G2 = nx.to_undirected(G)

  cliques = list(nx.find_cliques(G2))
  print(cliques)
  print(max( list(nx.find_cliques(G2)), key=lambda x:len(x)))

  # edges_tensor = torch.LongTensor([x for x in dataset[0][0]['edge_index']])
  # m = torch.LongTensor([x for x in dataset[0][1]])
  # print(utils.homophily_ratio(edges_tensor, m))

if __name__ == '__main__':
  # name = input('Choose OGB dataset node prediction [arxiv, products, proteins, mag, papers100M]: ')
  # name = 'ogbn-' + name
  # dataset = ogbn.NodePropPredDataset(name=name, root='/home/ctubert/tfg/gitprojects/gnn_analysis/analysis/datasets')
  # nodes_ini = list(range(0, dataset[0][0]['num_nodes']))
  # edges_ini = dataset[0][0]['edge_index']
  
  # edges_tensor = torch.LongTensor([x for x in edges_ini])
  # undirected = utils.is_undirected(edges_tensor)
  # print(undirected)

  # edge_list = []
  # for i in range(len(edges_ini[0])):
  #   edge_list.append((int(edges_ini[0][i]), int(edges_ini[1][i])))

  # if undirected:
  #   G = nx.Graph()
  # else:
  #   G = nx.DiGraph()
    
  # G.add_nodes_from(nodes_ini)
  # G.add_edges_from(edge_list)
  # print(nx.algorithms.cluster.transitivity(G))

  
  main()