import time
import pandas as pd
import numpy as np
import networkx as nx
#import cugraph as cnx
import random as rand
import csv

import ogb.nodeproppred as ogbn
from torch_geometric.data import Data
import torch_geometric.utils as utils
import torch_geometric
import torch

from torch_geometric.datasets import Planetoid

MAX_NODES = -1
TYPE_SPLIT = ''
TYPE_EXCEL = ''

values_dict = {}

def choose_dataset():
  data = input('Choose data [OGB, Planetoid]: ')
  if data == 'OGB':
    name = input('Choose OGB dataset node prediction [arxiv, products, proteins, mag, papers100M]: ')
    name = 'ogbn-' + name
    dataset = ogbn.NodePropPredDataset(name=name, root='/home/ctubert/tfg/gitprojects/gnn_analysis/analysis/datasets')

  elif data == 'Planetoid':
    name = input('Choose Planetoid dataset [Cora]: ')
    dataset = Planetoid(name=name, root='/home/ctubert/tfg/gitprojects/gnn_analysis/analysis/datasets')

  split = input('Choose dataset split [train, valid, test, no-split]: ')

  ini_dict(name, split)
  
  return (data, dataset, split)

def first_split(dataset, split):
  if(split == 'no-split'):
    nodes_ini = list(range(0, dataset[0][0]['num_nodes']))
  else:
    split_idx = ogb.get_idx_split()
    nodes_ini = split_idx[split]

    edges_ini = ogb[0][0]['edge_index']
    return (nodes_ini, edges_ini)

def second_split_and_shuffle(nodes_ini, edges_ini, i):
  if TYPE_SPLIT == 'random':
    rand.shuffle(nodes_ini)
    nodes = nodes_ini[0:MAX_NODES]

  elif TYPE_SPLIT == 'ordered':
    j = min( (i+MAX_NODES), len(nodes_ini) )
    nodes = nodes_ini[i:j]

  nodes_tensor = torch.LongTensor([x for x in nodes])
  edges_tensor = torch.LongTensor([x for x in edges_ini])

  edges, _ = utils.subgraph(nodes_tensor, edges_tensor, num_nodes=len(nodes_ini))

  return (nodes, edges)

def get_nx_graph(nodes, edges):
  undirected = utils.is_undirected(edges)

  edge_list = []
  for i in range(len(edges[0])):
    edge_list.append((int(edges[0][i]), int(edges[1][i])))

  if undirected:
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edge_list)

  else:
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edge_list)

  return (G, undirected)

def planetoid_to_nx(dataset):
  D = Data(dataset[0].x, dataset[0].edge_index, dataset[0].y)
  G = utils.to_networkx(D)

  undirected = not nx.is_directed(G)

  return G, undirected

def get_biggest_CC(G, undirected):
  if undirected:
    CC = [G.subgraph(c).copy() for c in sorted(nx.algorithms.components.connected_components(G), key=len, reverse=True)]
  else:
    CC = [G.subgraph(c).copy() for c in sorted(nx.algorithms.components.strongly_connected_components(G), key=len, reverse=True)]

  num_CC = len(CC)
  cc = CC[0]
  num_nodes = cc.number_of_nodes()
  num_edges = cc.number_of_edges()

  values_dict['Num CC'].append(num_CC)
  values_dict['BCC num nodes'].append(num_nodes)
  values_dict['BCC num edges'].append(num_edges)

  return cc

def graph_processing(G, undirected):
  print('Check directed G', nx.is_directed(G))
  num_nodes = G.number_of_nodes()
  num_edges = G.number_of_edges()

  degree_hist = nx.degree_histogram(G)
  degrees = [a*b for a, b in zip(degree_hist, range(0, len(degree_hist)))]
  avg_degree = sum(degrees) / num_nodes

  avg_clustering = nx.algorithms.cluster.average_clustering(G)

  density = nx.density(G)

  values_dict['Num nodes'].append(num_nodes)
  values_dict['Num edges'].append(num_edges)
  values_dict['Average degree'].append(avg_degree)
  values_dict['Average clustering'].append(avg_clustering)
  values_dict['Density'].append(density)

def CC_processing(cc, undirected):
  print('Check directed cc', nx.is_directed(cc))
  num_nodes = cc.number_of_nodes()
  num_edges = cc.number_of_edges()

  avg_path_length = nx.average_shortest_path_length(cc)
  diameter = nx.diameter(cc)
  radius = nx.radius(cc)
  node_connectivity = nx.algorithms.connectivity.connectivity.node_connectivity(cc)
  edge_connectivity = nx.algorithms.connectivity.connectivity.edge_connectivity(cc)

  values_dict['BCC num nodes'].append(num_nodes)
  print(num_nodes, num_edges)
  values_dict['BCC num edges'].append(num_edges)
  values_dict['BCC average path length'].append(avg_path_length)
  values_dict['BCC diameter'].append(diameter)
  values_dict['BCC radius'].append(radius)
  values_dict['BCC node connectivity'].append(node_connectivity)
  values_dict['BCC edge connectivity'].append(edge_connectivity)

def ini_dict(name, split):
  values_dict['Dataset name'] = name
  values_dict['Directed'] = -1
  values_dict['First split'] = split
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
  values_dict['Execution time'] = []

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
  with open('results_OGBN.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, values_dict.keys())
    w.writeheader()
    lim = len(values_dict['Execution time'])
    for i in range(lim):
      subdict = get_subdict(i)
      print(subdict)
      w.writerow(subdict)

def write_csv():
  with open('results_OGBN.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, values_dict.keys())
    w.writeheader()
    w.writerow(values_dict)

def analysis(data, dataset, split):
  if data == 'OGB': nodes_ini, edges_ini = first_split(dataset, split)

  if TYPE_SPLIT == 'random':
    lim = 5
    step = 1

  elif TYPE_SPLIT == 'ordered':
    lim = len(nodes_ini)
    step = MAX_NODES

  for i in range(0, lim, step):
    if data == 'Planetoid':
      G, undirected = planetoid_to_nx(dataset)

    else:
      nodes, edges = second_split_and_shuffle(nodes_ini, edges_ini, i)
      G, undirected = get_nx_graph(nodes, edges)

    values_dict['Directed'] = (not undirected)

    time_ini = time.time()

    graph_processing(G, undirected)
    cc = get_biggest_CC(G, undirected)
    CC_processing(cc, undirected)

    time_end = time.time() - time_ini

    values_dict['Execution time'].append(time_end)
  
  if TYPE_EXCEL == 'mean':
    mean_dict()
    write_csv()

  elif TYPE_EXCEL == 'all':
    write_csv_all()

def main():
  data, dataset, split = choose_dataset()

  global MAX_NODES 
  MAX_NODES = int(input('Choose MAX_NODES: '))
  global TYPE_SPLIT
  TYPE_SPLIT = input('Choose TYPE_SPLIT [random, ordered]: ')
  global TYPE_EXCEL
  TYPE_EXCEL = input('Choose TYPE_EXCEL [mean, all]: ')
  analysis(data, dataset, split)

def test():
    dataset = Planetoid(name='Cora', root='/home/ctubert/tfg/gitprojects/gnn_analysis/analysis/datasets')
    print(utils.homophily_ratio(dataset[0].edge_index, dataset[0].y))
    D = Data(dataset[0].x, dataset[0].edge_index, dataset[0].y)
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