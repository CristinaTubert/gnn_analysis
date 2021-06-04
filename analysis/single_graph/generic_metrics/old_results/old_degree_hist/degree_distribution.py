import time
import pandas as pd
import numpy as np
import networkx as nx
import random as rand
import csv
import collections
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

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
  ogb = ogbn.NodePropPredDataset(name=name, root='/home/ctubert/tfg/gitprojects/gnn_analysis/analysis/datasets')

  split = input('Choose dataset split [train, valid, test, no-split]: ')

  global MAX_NODES 
  text = 'Choose MAX_NODES [0, ' + str(ogb[0][0]['num_nodes']) + ']: '
  MAX_NODES = int(input(text))
  
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

def get_nx_graph(nodes, edges, edges_ini):
  edges_tensor = torch.LongTensor([x for x in edges_ini])
  undirected = utils.is_undirected(edges_tensor)

  edge_list = []
  for i in range(len(edges[0])):
    edge_list.append((int(edges[0][i]), int(edges[1][i])))

  edge_list_all = []
  for i in range(len(edges_ini[0])):
    edge_list_all.append((int(edges_ini[0][i]), int(edges_ini[1][i])))

  if undirected:
    G = nx.Graph()
    G_all = nx.Graph()

  else:
    G = nx.DiGraph()
    G_all = nx.DiGraph()
  
  G.add_nodes_from(nodes)
  G.add_edges_from(edge_list)
  G_all.add_nodes_from(nodes)
  G_all.add_edges_from(edge_list_all)

  return (G, G_all, undirected)

def histogram_degree(G, G_all, nodes, undirected, name, i):
  degree_sequence = sorted([d for n, d in G_all.degree(nodes)])  # degree sequence
  #take 90%
  print(degree_sequence)
  k = int(len(degree_sequence)*0.99)

  degreeCount = collections.Counter(degree_sequence[:k])
  deg, cnt = zip(*degreeCount.items())

  fig, ax = plt.subplots()
  plt.bar(deg, cnt, color="b")
  plt.xticks(rotation='vertical', fontsize=8)

  plt.title("Degree Histogram (0.99)")
  plt.ylabel("Count")
  plt.xlabel("Degree")

  '''
  plt.xticks([d + 0.4 for d in deg])
  ax.set_xticklabels([d for d in deg])
  '''

  ax.xaxis.set_major_locator(MultipleLocator(5))
  ax.xaxis.set_major_formatter('{x:.0f}')
  ax.xaxis.set_minor_locator(MultipleLocator(1))
  plt.xlim([0, max(deg)])


  ax.yaxis.set_major_locator(MultipleLocator(int(max(cnt)/20)))
  ax.yaxis.set_major_formatter('{x:.0f}')
  ax.yaxis.set_minor_locator(MultipleLocator(int(max(cnt)/100)))

  edge_cut = 0
  for n in nodes:
    rd = G_all.degree(n)
    d = G.degree(n)
    edge_cut += rd - d
    
  text = 'edge cuts = ' + str(edge_cut) + '\n' + 'max degree = ' + str(max(degree_sequence))
  plt.annotate(text, xy=(0.70, 0.90), xycoords='axes fraction')
  
  '''# draw graph in inset
  plt.axes([0.4, 0.4, 0.5, 0.5])
  Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
  pos = nx.spring_layout(G)
  plt.axis("off")
  nx.draw_networkx_nodes(G, pos, node_size=20)
  nx.draw_networkx_edges(G, pos, alpha=0.4)'''

  if (TYPE_SPLIT == 'random'):
    path = "./results/" + name + "/random/" "/histogram" + str(MAX_NODES) + "_" + str(i) + ".png" 
  
  else:
    path = "./results/" + name + "/histogram" + str(MAX_NODES) + "_" + str(i) + ".png"
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

    G, G_all, undirected = get_nx_graph(nodes, edges, edges_ini)

    histogram_degree(G, G_all, nodes, undirected, ogb.name, i)

def main():
  ogb, split = choose_node_dataset()

  global TYPE_SPLIT
  TYPE_SPLIT = input('Choose TYPE_SPLIT [random, ordered]: ')
  analysis(ogb, split)
    
def test():
  ogb = ogbn.NodePropPredDataset(name='ogbn-arxiv', root='../../datasets/')
  nodes_ini = list(range(0, ogb[0][0]['num_nodes']))
  edges_ini = ogb[0][0]['edge_index']
  rand.shuffle(nodes_ini)

  nodes = nodes_ini[0:1000]

  nodes_tensor = torch.LongTensor([x for x in nodes])
  edges_tensor = torch.LongTensor([x for x in edges_ini])

  edges, _ = utils.subgraph(nodes_tensor, edges_tensor, num_nodes=len(nodes_ini))

  time_ini = time.time()
  edge_list = []
  for i in range(len(edges_ini[0])):
    if (int(edges_ini[0][i]) in nodes) or (int(edges_ini[1][i]) in nodes):
      edge_list.append((int(edges_ini[0][i]), int(edges_ini[1][i])))

  G = nx.DiGraph()
  G.add_nodes_from(nodes)
  G.add_edges_from(edge_list)
  print(G.number_of_nodes())
  G.degree(nodes)

  time_end = time.time() - time_ini
  print(time_end)

  time_ini2 = time.time()
  
  edge_list2 = []
  for i in range(len(edges_ini[0])):
    edge_list2.append((int(edges_ini[0][i]), int(edges_ini[1][i])))

  X = nx.DiGraph()
  X.add_nodes_from(nodes)
  X.add_edges_from(edge_list2)
  print(X.number_of_nodes())
  X.degree(nodes)

  time_end2 = time.time() - time_ini2
  print(time_end2)
 
if __name__ == '__main__':
  main()