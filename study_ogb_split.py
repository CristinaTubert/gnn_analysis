import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import ogb.nodeproppred as ogbn
import torch_geometric
from torch_geometric.data import Data
import torch_geometric.utils as utils
import torch

### OGB DATASETS NODE PREDICTION
#products = ogbn.NodePropPredDataset(name='ogbn-products', root='dataset/')
#proteins = ogbn.NodePropPredDataset(name='ogbn-proteins', root='dataset/')
#papers = ogbn.NodePropPredDataset(name='ogbn-papers100M', root='dataset/')
#mag = ogbn.NodePropPredDataset(name='ogbn-mag', root='dataset/')

def ogb_to_first_split(ogb_dataset, split_type):

  #first split dataset
  split_idx = ogb_dataset.get_idx_split()
  partial_dataset = split_idx[split_type]

  #induced graph containing subset nodes
  edge_index_tensor = torch.LongTensor([x for x in ogb_dataset[0][0]['edge_index']])
  partial_nodes_tensor = torch.LongTensor([x for x in partial_dataset])
  edge_index, _ = utils.subgraph(partial_nodes_tensor, edge_index_tensor)

  #convert to networkx graph
  edge_list = []
  for i in range(len(edge_index[0])):
    edge_list.append((int(edge_index[0][i]), int(edge_index[1][i])))

  print('FIRST SPLIT INFORMATION')

  print('Split type:', split_type)

  G = nx.to_networkx_graph(edge_list)
  print('Number of nodes first split:', G.number_of_nodes())
  print('Number of edges first split:', G.number_of_edges())
  
  return edge_list

def do_sub_splits(edge_list, mnodes):
  i = 0
  nsplit = 1

  while i < (len(edge_list) + mnodes):
    j = i + mnodes-1
    G = nx.to_networkx_graph(edge_list[i:j]) #G might not be fully connected

    print('i =', i, 'j=', j)
    cc = biggest_connected_subraph(G, nsplit)
    if (cc==None): break
    extract_features(cc, nsplit)

    i = i + mnodes
    nsplit = nsplit+1


def biggest_connected_subraph(G, nsplit):
  #generate a sorted list of connected components, largest first
  cc = [G.subgraph(c).copy() for c in sorted(nx.algorithms.components.connected_components(G), key=len, reverse=True)]
  if (len(cc)==0): return None

  print('SUB-SPLIT INFORMATION')

  print('Split number:', str(nsplit))
  print('Number of connected subgraphs:', len(cc))
  print('Number of nodes on the biggest subgraph:', len(cc[0]))
  print('Number of nodes on the second biggest subgraph:', len(cc[1]))

  print()

  return cc[0]

def extract_features(G, nsplit):

  print('BIGGEST CONNECTED SUBGRAPH INFORMATION')

  print('Number of nodes:', G.number_of_nodes())
  print('Number of edges:', G.number_of_edges())

  G_deg = nx.degree_histogram(G)
  plt.plot(G_deg)
  path = "./degree_histograms/histogram" + str(nsplit) + ".png"
  plt.savefig(path)

  G_deg_sum = [a * b for a, b in zip(G_deg, range(0, len(G_deg)))]
  average_degree = sum(G_deg_sum) / G.number_of_nodes()
  print('Average degree:', average_degree)

  if nx.is_connected(G):
    print('Average path length:', nx.average_shortest_path_length(G))
    print('Average diameter:', nx.diameter(G))

  G_cluster = sorted(list(nx.clustering(G).values()))
  average_cluster_coef = sum(G_cluster) / len(G_cluster)
  print('Average clustering coefficient:', average_cluster_coef)

  print()

### MAIN
arxiv = ogbn.NodePropPredDataset(name='ogbn-arxiv', root='dataset/')
max_num_nodes = 2000
print("Max number of nodes:", max_num_nodes)

#split_list = ["train", "valid", "test"]
split_list = ["valid"]

for split in split_list:
  edge_list = ogb_to_first_split(arxiv, split)
  do_sub_splits(edge_list, max_num_nodes)
  
  


