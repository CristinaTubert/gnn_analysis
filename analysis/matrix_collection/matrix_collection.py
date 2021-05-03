import time
import pandas as pd
import numpy as np
import networkx as nx
import random as rand
import csv
import scipy.io
import matplotlib.pyplot as plt
import os.path
#import cugraph as cnx

MAX_NODES = -1
TYPE_SPLIT = ''
TYPE_EXCEL = ''

values_dict = {}

def mat_to_nx():
  name = input('Choose matrix name : ')
  path = 'datasets/' + name + '.mat'
  mat = scipy.io.loadmat(path)
  mat = mat['Problem']

  for el in mat[0][0]:
    if type(el) == scipy.sparse.csc.csc_matrix:
      break

  matrix = el.toarray()
  G = nx.convert_matrix.from_numpy_matrix(matrix)

  ini_dict(name)
  
  return G

def choose_globals(num_nodes):
  global MAX_NODES 
  text = 'Choose MAX_NODES [0, ' + str(num_nodes) + ']: '
  MAX_NODES = int(input(text))

  global TYPE_EXCEL
  TYPE_EXCEL = input('Choose TYPE_EXCEL [mean, all]: ')

  global TYPE_SPLIT
  TYPE_SPLIT = input('Choose TYPE_SPLIT [random, ordered]: ')

def split(G, i):
  nodes_ini = list(range(0, G.number_of_nodes()))

  if TYPE_SPLIT == 'random':
    rand.shuffle(nodes_ini)
    nodes = nodes_ini[0:MAX_NODES]

  elif TYPE_SPLIT == 'ordered':
    j = min( (i+MAX_NODES), len(nodes_ini) )
    nodes = nodes_ini[i:j]

  subG = G.subgraph(nodes)
  undirected = not nx.is_directed(subG)

  return subG, undirected

def get_biggest_CC(G, undirected):
  if undirected:
    CC = [G.subgraph(c).copy() for c in sorted(nx.algorithms.components.connected_components(G), key=len, reverse=True)]
  else:
    CC = [G.subgraph(c).copy() for c in sorted(nx.algorithms.components.strongly_connected_components(G), key=len, reverse=True)]

  num_CC = len(CC)
  cc = CC[0]
  
  values_dict['Num CC'].append(num_CC)

  return cc

def graph_processing(G, undirected):
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
  num_nodes = cc.number_of_nodes()
  num_edges = cc.number_of_edges()

  avg_path_length = nx.average_shortest_path_length(cc)
  diameter = nx.diameter(cc)
  radius = nx.radius(cc)
  node_connectivity = nx.algorithms.connectivity.connectivity.node_connectivity(cc)
  edge_connectivity = nx.algorithms.connectivity.connectivity.edge_connectivity(cc)

  values_dict['BCC num nodes'].append(num_nodes)
  values_dict['BCC num edges'].append(num_edges)
  values_dict['BCC average path length'].append(avg_path_length)
  values_dict['BCC diameter'].append(diameter)
  values_dict['BCC radius'].append(radius)
  values_dict['BCC node connectivity'].append(node_connectivity)
  values_dict['BCC edge connectivity'].append(edge_connectivity)

def ini_dict(name):
  values_dict['Dataset name'] = name
  values_dict['Directed'] = -1
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
  file_name = values_dict['Dataset name'] + '.csv'
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


def graph_to_characterization(preG, i):
  G, undirected = split(preG, i)

  values_dict['Directed'] = (not undirected)

  time_ini = time.time()

  graph_processing(G, undirected)
  cc = get_biggest_CC(G, undirected)
  CC_processing(cc, undirected)

  time_end = time.time() - time_ini

  values_dict['Execution time'].append(time_end)

def analysis(G):
  choose_globals(G.number_of_nodes())

  if TYPE_SPLIT == 'random': 
    lim = int(input('Choose number of iterations for random splits: '))
    step = 1
    values_dict['Type split'] = str(lim) + ' random iterations'

  elif TYPE_SPLIT == 'ordered':
    lim = len(nodes_ini)
    step = MAX_NODES
    values_dict['Type split'] = 'ordered'
    
  for i in range(0, lim, step):
    print(i)
    graph_to_characterization(G, i)
  
  if TYPE_EXCEL == 'mean':
    mean_dict()
    write_csv_mean()
  elif TYPE_EXCEL == 'all':
    write_csv_all()

def main():
  G = mat_to_nx()
  analysis(G)

def explore():
  name = 'mhd1280b'
  path = 'datasets/' + name + '.mat'
  mat = scipy.io.loadmat(path)
  print(mat)
  print('\n')
  mat = mat['Problem']
  print(mat)
  for el in mat[0][0]:
    print(el)

 
if __name__ == '__main__':
  main()