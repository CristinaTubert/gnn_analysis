# -*- coding: utf-8 -*-
"""OGB.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16HSK1637zMyd2mh2GvluW2xLrLD13Ldp
"""

"""!pip install ogb
!pip install -U ogb
!pip install torch_geometric
!pip install torch_sparse
!pip install torch_scatter"""

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ogb.nodeproppred as ogbn
import ogb.graphproppred as ogbg
from torch_geometric.data import DataLoader
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data

#products = ogbn.NodePropPredDataset(name='ogbn-products', root='dataset/')
#proteins = ogbn.NodePropPredDataset(name='ogbn-proteins', root='dataset/')
#papers = ogbn.NodePropPredDataset(name='ogbn-papers100M', root='dataset/')
#mag = ogbn.NodePropPredDataset(name='ogbn-mag', root='dataset/')

arxiv = ogbn.NodePropPredDataset(name='ogbn-arxiv', root='dataset/')
split_idx = arxiv.get_idx_split()
partial_arxiv = split_idx["valid"]

def ogb_to_graph(ogb, partial_ogb):

  edge_index_tensor = torch.LongTensor([x for x in ogb[0][0]['edge_index']])
  subnodes_tensor = torch.LongTensor([x for x in partial_ogb])
  subG = utils.subgraph(subnodes_tensor, edge_index_tensor)

  edge_index = subG[0]

  edge_list = []
  for i in range(len(edge_index[0])):
    edge_list.append((edge_index[0][i], edge_index[1][i]))

  graph = nx.to_networkx_graph(edge_list)
  return graph

G = ogb_to_graph(arxiv, partial_arxiv)

print('num of nodes: {}'.format(G.number_of_nodes()))

print('num of edges: {}'.format(G.number_of_edges()))

G_deg = nx.degree_histogram(G)
G_deg_sum = [a * b for a, b in zip(G_deg, range(0, len(G_deg)))]
print('average degree: {}'.format(sum(G_deg_sum) / G.number_of_nodes()))

if nx.is_connected(G):
    print('average path length: {}'.format(nx.average_shortest_path_length(G)))
    print('average diameter: {}'.format(nx.diameter(G)))

G_cluster = sorted(list(nx.clustering(G).values()))
print('average clustering coefficient: {}'.format(sum(G_cluster) / len(G_cluster)))