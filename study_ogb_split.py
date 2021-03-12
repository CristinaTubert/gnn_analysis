import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time

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

def ogb_to_subgraphs(ogb_dataset, split_type):

  #first split dataset
  split_idx = ogb_dataset.get_idx_split()
  partial_dataset = split_idx[split_type]

  print('FIRST SPLIT INFORMATION')
  print('Split type:', split_type)
  print('Number of nodes first split:', len(partial_dataset), '\n')

  #second splits dataset
  i = 0
  nsplit = 1
  lsubG = []

  while i < len(partial_dataset):
    j = min((i + MAXNODES), len(partial_dataset))
    nodes_subset = partial_dataset[i:j]

    #induced subgraph containing subset nodes
    edge_index_tensor = torch.LongTensor([x for x in ogb_dataset[0][0]['edge_index']])
    nodes_subset_tensor = torch.LongTensor([x for x in nodes_subset])
    edge_index, _ = utils.subgraph(nodes_subset_tensor, edge_index_tensor)

    #convert to networkx graph
    edge_list = []
    for k in range(len(edge_index[0])):
      edge_list.append((int(edge_index[0][k]), int(edge_index[1][k])))

    G = nx.to_networkx_graph(edge_list)
    lsubG.append((nsplit, G))

    i = i + MAXNODES
    nsplit = nsplit+1
  
  return lsubG

def biggest_connected_subraph(nsplit, G):
  #generate a sorted list of connected components, largest first
  cc = [G.subgraph(c).copy() for c in sorted(nx.algorithms.components.connected_components(G), key=len, reverse=True)]
  if (len(cc)==0): return None

  print('SUB-SPLIT INFORMATION')

  print('Split number:', str(nsplit))
  print('Number of connected subgraphs:', len(cc))
  print('Number of nodes on the biggest subgraph:', len(cc[0]))
  print('Number of nodes on the second biggest subgraph:', len(cc[1]))

  return cc[0]

def extract_features(nsplit, G):

  print('BIGGEST CONNECTED SUBGRAPH INFORMATION')

  print('Number of nodes:', G.number_of_nodes())
  print('Number of edges:', G.number_of_edges())

  G_deg = nx.degree_histogram(G)
  plt.plot(G_deg)
  plt.xlabel("Node degree")
  path = "./degree_histograms/histogram" + str(nsplit) + ".png"
  plt.savefig(path)
  plt.clf()

  G_deg_sum = [a * b for a, b in zip(G_deg, range(0, len(G_deg)))]
  average_degree = sum(G_deg_sum) / G.number_of_nodes()
  print('Average degree:', average_degree)

  if nx.is_connected(G):
    print('Average path length:', nx.average_shortest_path_length(G))
    print('Average diameter:', nx.diameter(G))

  G_cluster = sorted(list(nx.clustering(G).values()))
  average_cluster_coef = sum(G_cluster) / len(G_cluster)
  print('Average clustering coefficient:', average_cluster_coef, '\n')

### MAIN
start_time = time.time()

#arxiv = ogbn.NodePropPredDataset(name='ogbn-arxiv', root='dataset/')
proteins = ogbn.NodePropPredDataset(name='ogbn-proteins', root='dataset/')

MAXNODES = int(input())
split = str(input())
print("Max number of nodes:", MAXNODES, '\n')

lsubG = ogb_to_subgraphs(proteins, split)
for nsplit,subG in lsubG:
  cc = biggest_connected_subraph(nsplit, subG)
  if (cc==None): break
  extract_features(nsplit,cc)

print("Execution time: %s seconds" % (time.time() - start_time))


  
  


