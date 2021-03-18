import time
import pandas as pd
import numpy as np
import networkx as nx
import random as rand
import csv

import ogb.nodeproppred as ogbn
from torch_geometric.data import Data
import torch_geometric.utils as utils
import torch_geometric
import torch

global MAX_NODES = -1

values_dict = {}

def choose_node_dataset():
  name = input('Choose dataset node prediction [arxiv, products, proteins, mag, papers100M]: ')
  name = 'ogbn-' + name
  ogb = ogbn.NodePropPredDataset(name=name, root='dataset/')

  split = input('Choose dataset split [train, valid, test, no-split]: ')
  
  return (ogb, split)

def choose_graph_dataset():
  return (None, None)

def first_split(ogb, split):
  if(split == 'no-split'):
    nodes_ini = list(range(0, ogbDataset[0][0]['num_nodes']))

  else:
    split_idx = ogb.get_idx_split()
    nodes_ini = split_idx[split]

  edges_ini = ogb[0][0]['edge_index']
  return (nodes_ini, edges_ini)

def second_split_and_shuffle(nodes_ini, edges_ini):
  rand.shuffle(nodes_ini)
  print(nodes_ini)
  nodes = nodes_ini[0:MAX_NODES]
  print(nodes)
  print(edges_ini)
  nodes_tensor = torch.LongTensor([x for x in nodes])
  edges_tensor = torch.LongTensor([x for x in edges_ini])
  print(edges_tensor)
  edges, _ = utils.subgraph(nodes_tensor, edges_tensor)

  return (nodes, edges)

def get_nx_graph(nodes, edges):
  print(edges)
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

  values_dict['BCC num nodes'].append(num_nodes)
  values_dict['BCC num edges'].append(num_edges)
  values_dict['BCC average path length'].append(avg_path_length)
  values_dict['BCC diameter'].append(diameter)
  values_dict['BCC radius'].append(radius)

def ini_dict(task, name, split):
  values_dict['Task'] = task
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
  values_dict['Execution time'] = []

def mean_dict():
  for key,value in values_dict.items():
    if type(value) == list:
      values_dict[key] = sum(value)/len(value)

def write_csv():
  with open('results.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, values_dict.keys())
    w.writeheader()
    w.writerow(values_dict)

def node_pred_analysis(ogb, split):
  ini_dict('node', ogb.name, split)

  nodes_ini, edges_ini = first_split(ogb, split)
  print(edges_ini)
  print(nodes_ini)

  for i in range(5):
    time_ini = time.time()

    nodes, edges = second_split_and_shuffle(nodes_ini, edges_ini)
    print(edges)

    values_dict['Num nodes'].append(len(nodes))
    values_dict['Num edges'].append(len(edges))

    G, undirected = get_nx_graph(nodes, edges)

    values_dict['Directed'] = (not undirected)

    graph_processing(G, undirected)

    if (undirected and not nx.is_connected(G)) or (not undirected and not nx.is_strongly_connected(G)):
      cc = get_biggest_CC(G, undirected)

    CC_processing(cc, undirected)

    time_end = time.time() - time_ini

    values_dict['Execution time'].append(time_end)
  
  mean_dict()
  write_csv()

def main():
  task = input('Choose dataset task prediction [nodepred, graphpred]: ')
  
  if task == 'nodepred':
    ogb, split = choose_node_dataset()
    MAX_NODES = int(input('Choose MAX_NODES: '))
    print(type(MAX_NODES))
    node_pred_analysis(ogb, split)
    
  elif task == 'graphpred':
    ogb, split = choose_graph_dataset()
    graph_pred_analysis(ogb, split)

if __name__ == '__main__':
  main()