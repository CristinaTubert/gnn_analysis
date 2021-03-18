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

NUM_GRAPHS = -1

values_dict = {}

def choose_graph_dataset():
    name = input('Choose dataset node prediction [molhiv, molpcba, ppa, code]: ')
    name = 'ogbg-' + name
    ogb = ogbn.NodePropPredDataset(name=name, root='dataset/')
  
    return (ogb, split)

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

def graph_processing(G, undirected):
    print('Check directed G', nx.is_directed(G))

    ini_dict(ogb.name, split)

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    degree_hist = nx.degree_histogram(G)
    degrees = [a*b for a, b in zip(degree_hist, range(0, len(degree_hist)))]
    avg_degree = sum(degrees) / num_nodes

    avg_clustering = nx.algorithms.cluster.average_clustering(G)

    density = nx.density(G)

    avg_path_length = nx.average_shortest_path_length(cc)

    diameter = nx.diameter(cc)

    radius = nx.radius(cc)

    time_end = time.time() - time_ini

    values_dict['Directed'] = (not undirected)
    values_dict['Num nodes'] = num_nodes
    values_dict['Num edges'] = num_edges
    values_dict['Average degree'] = avg_degree
    values_dict['Average clustering'] = avg_clustering
    values_dict['Density'] = density
    values_dict['Average path length'] = avg_path_length
    values_dict['Diameter'] = diameter
    values_dict['Radius'] = radius
    values_dict['Execution time'] = time_end

def write_csv():
    with open('results_OGBG.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, values_dict.keys())
        w.writeheader()
        w.writerow(values_dict)

def analysis(ogb):
    f = open('results_OGBG.csv', 'w', newline='')
    w = csv.writer(f)
    write_header = True

    for i in range(NUM_GRAPHS):
        values_dict = {}

        graph_id = rand.randint(0, len(ogb))
        values_dict['Dataset name'] = ogb.name
        values_dict['Graph ID'] = graph_id

        ogb_graph = ogb[graph_id]

        nodes = list(range(0, ogb_graph[0]['num_nodes']))
        edges = ogb_graph[0]['edge_index']

        nodes_tensor = torch.LongTensor([x for x in nodes])
        edges_tensor = torch.LongTensor([x for x in edges])
        edges, _ = utils.subgraph(nodes_tensor, edges_tensor)

        G, undirected = get_nx_graph(nodes, edges)
        graph_processing(G, undirected)

        if write_header:
            w.writerow(values_dict.keys())
            write_header = False

        w.writerow(values_dict.values())

def main():
 
    ogb = choose_graph_dataset()
    global NUM_GRAPHS
    NUM_NODES = int(input('Choose NUM_GRAPHS: '))
    analysis(ogb)

if __name__ == '__main__':
  main()