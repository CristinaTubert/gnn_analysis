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

MAX_NODES = -1

values_dict = {}

def choose_node_dataset():
  name = 'arxiv'
  name = 'ogbn-' + name
  ogb = ogbn.NodePropPredDataset(name=name, root='dataset/')

  split = 'train'
  
  return (ogb, split)

def first_split(ogb, split):
  total_num_nodes = ogb[0][0]['num_nodes']

  if(split == 'no-split'):
    nodes_ini = list(range(0, ogb[0][0]['num_nodes']))

  else:
    split_idx = ogb.get_idx_split()
    nodes_ini = split_idx[split]

  edges_ini = ogb[0][0]['edge_index']
  
  print(ogb[0][0])
  feature = input('Choose feature: ')
  node_feat = ogb[0][0][feature]

  return (total_num_nodes, nodes_ini, edges_ini, node_feat)

def second_split_and_shuffle(total_num_nodes, nodes_ini, edges_ini):
  rand.shuffle(nodes_ini)
  nodes = nodes_ini[0:MAX_NODES]

  nodes_tensor = torch.LongTensor([x for x in nodes])
  edges_tensor = torch.LongTensor([x for x in edges_ini])

  edges, _ = utils.subgraph(nodes_tensor, edges_tensor, num_nodes=total_num_nodes)

  return (nodes, edges)

def get_dict_features(features):
    dict_features = {}
    for i in range(len(features)):
        dict_feature = {str(j): features[i][j] for j in range(0, len(features[i]))}
        dict_features[i] = dict_feature
    
    return dict_features

def get_nx_graph(nodes, edges, features):
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

  dict_features = get_dict_features(features)
  nx.set_node_attributes(G, dict_features)

  CC = [G.subgraph(c).copy() for c in sorted(nx.algorithms.components.strongly_connected_components(G), key=len, reverse=True)]
  print(len(CC))

  return (G, undirected)

def compute_assortativity(G, num_features):
  for i in range(num_features):
    assortativity = nx.attribute_assortativity_coefficient(G, str(i))
    value = 'Assortativity_' + str(i+1)
    values_dict[value] = assortativity

def write_csv(i):
  if i==0:
    with open('results_OGBN.csv', 'w', newline='') as f:
      w = csv.DictWriter(f, values_dict.keys())
      w.writeheader()
      w.writerow(values_dict)

  else:
    with open('results_OGBN.csv', 'a', newline='') as f:
      w = csv.DictWriter(f, values_dict.keys())
      w.writerow(values_dict)

def analysis(ogb, split):

  total_num_nodes, nodes_ini, edges_ini, node_feat = first_split(ogb, split)
  for i in range(5):
    nodes, edges = second_split_and_shuffle(total_num_nodes, nodes_ini, edges_ini)

    G, undirected = get_nx_graph(nodes, edges, node_feat)

    values_dict['Directed'] = (not undirected)

    time_ini = time.time()

    num_features = len(node_feat[0])
    compute_assortativity(G, num_features)

    time_end = time.time() - time_ini

    values_dict['Execution time'] = time_end
  
    write_csv(i)

def main():
  ogb, split = choose_node_dataset()

  global MAX_NODES 
  MAX_NODES = int(input('Choose MAX_NODES: '))
  analysis(ogb, split)

def test():
    nodes = [0,3,5]
    edge_list = [(0,5), (3,5)]
    features = [[0.02,0.003], [0.07,0.2], [0.001,0.2], [0.001,0.02], [0.1,0.05], [0.01,0.09]]
    dict_features = {}
    for i in range(len(features)):
        dict_feature = {str(j): features[i][j] for j in range(0, len(features[i]))}
        dict_features[i] = dict_feature
        print(dict_features)

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edge_list)

    nx.set_node_attributes(G,dict_features)

    R = nx.Graph()
    R.add_node(4, color=1, dim=2)
    R.add_node(5, color=67, dim=9)

    for i in range(2):
      assortativity = nx.attribute_assortativity_coefficient(G, str(i))
      print(assortativity)

 
if __name__ == '__main__':
  main()