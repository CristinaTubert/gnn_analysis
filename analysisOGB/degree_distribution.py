import time
import pandas as pd
import numpy as np
import networkx as nx
import random as rand
import csv
import collections
import matplotlib.pyplot as plt

import ogb.nodeproppred as ogbn
from torch_geometric.data import Data
import torch_geometric.utils as utils
import torch_geometric
import torch

MAX_NODES = -1
TYPE_SPLIT = ''

def choose_node_dataset():
  name = input('Choose dataset node prediction [arxiv, products, proteins, mag, papers100M]: ')
  name = 'ogbn-' + name
  ogb = ogbn.NodePropPredDataset(name=name, root='dataset/')

  split = input('Choose dataset split [train, valid, test, no-split]: ')
  
  return (ogb, split)

def first_split(ogb, split):
  if(split == 'no-split'):
    nodes_ini = list(range(0, ogb[0][0]['num_nodes']))

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

def histogram_degree(G, undirected, name, i):
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())

    fig, ax = plt.subplots()
    plt.bar(deg, cnt, width=0.80, color="b")
    plt.xticks(rotation='vertical')

    plt.title("Degree Histogram")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    ax.set_xticks([d + 0.4 for d in deg])
    ax.set_xticklabels(deg)

    '''# draw graph in inset
    plt.axes([0.4, 0.4, 0.5, 0.5])
    Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
    pos = nx.spring_layout(G)
    plt.axis("off")
    nx.draw_networkx_nodes(G, pos, node_size=20)
    nx.draw_networkx_edges(G, pos, alpha=0.4)'''

    path = "./degree_histograms/" + name + "/histogram" + str(MAX_NODES) + "_" + str(i) + ".png"
    plt.savefig(path)
    plt.clf()

def analysis(ogb, split):
  nodes_ini, edges_ini = first_split(ogb, split)

  if TYPE_SPLIT == 'random':
    lim = 5
    step = 1

  elif TYPE_SPLIT == 'ordered':
    lim = len(nodes_ini)
    step = MAX_NODES

  for i in range(0, lim, step):
    nodes, edges = second_split_and_shuffle(nodes_ini, edges_ini, i)

    G, undirected = get_nx_graph(nodes, edges)

    histogram_degree(G, undirected, ogb.name, i)

def main():
  ogb, split = choose_node_dataset()

  global MAX_NODES 
  MAX_NODES = int(input('Choose MAX_NODES: '))
  global TYPE_SPLIT
  TYPE_SPLIT = input('Choose TYPE_SPLIT [random, ordered]: ')
  analysis(ogb, split)
    
 
if __name__ == '__main__':
  main()