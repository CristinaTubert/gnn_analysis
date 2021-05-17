import time
import pandas as pd
import numpy as np
import networkx as nx
import random as rand
import csv
import os.path

#import cugraph as cnx

import ogb.nodeproppred as ogbn
from torch_geometric.data import Data
import torch_geometric.utils as utils
import torch_geometric
import torch

from torch_geometric.datasets import Planetoid

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
    name = input('Choose Planetoid dataset [Cora, CiteSeer]: ')
    dataset = Planetoid(name=name, root='/home/ctubert/tfg/gitprojects/gnn_analysis/analysis/datasets')
    fsplit = 'no-split'

  ini_dict(fsplit, name)
  
  return (data, dataset, fsplit)

def choose_globals(num_nodes):
  global SIZE_SPLIT 
  text = 'Choose SIZE_SPLIT [0, ' + str(num_nodes) + ']: '
  SIZE_SPLIT = int(input(text))

  global METRICS_RESULTS
  METRICS_RESULTS = input('Choose METRICS_RESULTS [mean, all]: ')

  global TYPE_SPLIT
  TYPE_SPLIT = input('Choose TYPE_SPLIT [random, ordered]: ')

def first_split(data, dataset, fsplit):
  if data == 'OGB':
    if fsplit == 'no-split':
      nodes_ini = list(range(0, dataset[0][0]['num_nodes']))
    else:
      split_idx = dataset.get_idx_split()
      nodes_ini = split_idx[fsplit]

    edges_ini = dataset[0][0]['edge_index']

    features = dataset[0][1]

  elif data == 'Plan':
    nodes_ini = list(range(0, len(dataset[0].y))) #always no split
    edges_ini = -1 #no matter

    features = dataset[0].y
    
  name_features = 'y'
  values_dict['Feature studied'] = name_features

  return (nodes_ini, edges_ini, features, name_features)

def second_split_OGB(nodes_ini, edges_ini, i):
  if TYPE_SPLIT == 'random':
    rand.shuffle(nodes_ini)
    nodes = nodes_ini[0:SIZE_SPLIT]

  elif TYPE_SPLIT == 'ordered':
    j = min( (i+SIZE_SPLIT), len(nodes_ini) )
    nodes = nodes_ini[i:j]

  nodes_tensor = torch.LongTensor([x for x in nodes])
  edges_tensor = torch.LongTensor([x for x in edges_ini])

  edges, _ = utils.subgraph(nodes_tensor, edges_tensor, num_nodes=len(nodes_ini))

  return nodes, edges

def second_split_planetoid(preG, nodes_ini, i):
  if TYPE_SPLIT == 'random':
    rand.shuffle(nodes_ini)
    nodes = nodes_ini[0:SIZE_SPLIT]

  elif TYPE_SPLIT == 'ordered':
    j = min( (i+SIZE_SPLIT), len(nodes_ini) )
    nodes = nodes_ini[i:j]

  G = preG.subgraph(nodes)
  return G

def OGB_to_nx(nodes, edges, features_n, name_features):
  undirected = utils.is_undirected(edges)

  edge_list = []
  for i in range(len(edges[0])):
    edge_list.append((int(edges[0][i]), int(edges[1][i])))

  if undirected:
    G = nx.Graph()
  else:
    G = nx.DiGraph()

  G.add_nodes_from(nodes)
  G.add_edges_from(edge_list)

  #print(features_n)
  features = [int(f) for f in features_n]
  dict_features = list_to_dict(features)
  #print(dict_features)
  nx.set_node_attributes(G, dict_features, name_features) #controlar que es una lista de ints

  return (G, undirected)

def planetoid_to_nx(dataset, features_t, name_features):
  #print(features_t)
  D = Data(dataset[0].x, dataset[0].edge_index, dataset[0].y)
  #print(dataset[0].edge_index)
  G = utils.to_networkx(D) #to_undirected, default always directed

  features = [int(f) for f in features_t]
  dict_features = list_to_dict(features)
  nx.set_node_attributes(G, dict_features, name_features)
  undirected = not nx.is_directed(G)

  return G, undirected

def get_biggest_CC(G, undirected):
  if undirected:
    CC = [G.subgraph(c) for c in sorted(nx.algorithms.components.connected_components(G), key=len, reverse=True)]
  else:
    CC = [G.subgraph(c) for c in sorted(nx.algorithms.components.strongly_connected_components(G), key=len, reverse=True)]

  num_CC = len(CC)
  cc = CC[0]

  values_dict['Num CC'].append(num_CC)
  values_dict['BCC num nodes'].append(cc.number_of_nodes())
  values_dict['BCC num edges'].append(cc.number_of_edges())

  return cc

def list_to_dict(l):
  nums = list(range(0,len(l)))
  dictionary = dict(zip(nums,l))

  return dictionary

def ini_dict(fsplit, name):
  values_dict['Dataset name'] = name
  values_dict['Feature studied'] = ''
  values_dict['Directed'] = -1
  values_dict['First split'] = fsplit
  values_dict['Num nodes'] = []
  values_dict['Num edges'] = []
  values_dict['Homophily'] = []
  values_dict['Num CC'] = []
  values_dict['BCC num nodes'] = []
  values_dict['BCC num edges'] = []
  values_dict['BCC homophily'] = []
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

def compute_assortativity(G, undirected, name_features):

  time_ini = time.time()

  assortativity = nx.attribute_assortativity_coefficient(G, name_features)
  cc = get_biggest_CC(G, undirected)
  cc_assortativity = nx.attribute_assortativity_coefficient(cc, name_features)
  
  time_end = time.time() - time_ini

  values_dict['Homophily'].append(assortativity)
  values_dict['BCC homophily'].append(cc_assortativity)
  values_dict['Execution time'].append(time_end)

def split_control(data, dataset, fsplit):
  nodes_ini, edges_ini, features, name_features = first_split(data, dataset, fsplit)
  choose_globals(len(nodes_ini))

  if TYPE_SPLIT == 'random': 
    lim = int(input('Choose number of iterations for random splits: '))
    step = 1
    values_dict['Type split'] = str(lim) + ' random iterations'

  elif TYPE_SPLIT == 'ordered':
    lim = len(nodes_ini)
    total_it = len(nodes_ini)//SIZE_SPLIT
    num_it = int(input('Choose number of iterations for ordered splits [1, ' + str(total_it) + ']: '))
    step = (total_it//num_it)*SIZE_SPLIT
    print(step)
    values_dict['Type split'] = str(num_it) + ' ordered iterations'

  if data == 'Plan':
    preG, undirected = planetoid_to_nx(dataset, features, name_features)
    
  for i in range(0, lim, step):
    print('Iteration ' + str(i))

    if data == 'OGB':
      nodes, edges = second_split_OGB(nodes_ini, edges_ini, i)
      G, undirected = OGB_to_nx(nodes, edges, features, name_features)
    elif data == 'Plan':
      G = second_split_planetoid(preG, nodes_ini, i)
    
    #print(nx.get_node_attributes(G, 'y'))
    values_dict['Directed'] = (not undirected)
    values_dict['Num nodes'].append(G.number_of_nodes())
    values_dict['Num edges'].append(G.number_of_edges())
    
    compute_assortativity(G, undirected, name_features)
  
  print('Writing metrics...')
  if METRICS_RESULTS == 'mean':
    mean_dict()
    write_csv_mean()
    
  elif METRICS_RESULTS == 'all':
    write_csv_all()

def main():
  data, dataset, fsplit = choose_dataset()
  split_control(data, dataset, fsplit)



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
  dataset = Planetoid(name='CiteSeer', root='/home/ctubert/tfg/gitprojects/gnn_analysis/analysis/datasets')
  print(utils.homophily_ratio(dataset[0].edge_index, dataset[0].y))
  D = Data(dataset[0].x, dataset[0].edge_index, dataset[0].y)
  print(dataset[0])
  G = utils.to_networkx(D)
  print(G.number_of_nodes())

  name = input('Choose OGB dataset node prediction [arxiv, products, proteins, mag, papers100M]: ')
  name = 'ogbn-' + name
  dataset = ogbn.NodePropPredDataset(name=name, root='/home/ctubert/tfg/gitprojects/gnn_analysis/analysis/datasets')
  #print(dataset[0])

  edges_tensor = torch.LongTensor([x for x in dataset[0][0]['edge_index']])
  m = torch.LongTensor([x for x in dataset[0][1]])
  print(utils.homophily_ratio(edges_tensor, m))

if __name__ == '__main__':
  main()