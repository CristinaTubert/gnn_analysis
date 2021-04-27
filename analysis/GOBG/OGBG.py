import time
import pandas as pd
import numpy as np
import networkx as nx
import random as rand
import csv

import ogb.graphproppred as ogbg
from torch_geometric.data import Data
import torch_geometric.utils as utils
import torch_geometric
import torch

NUM_GRAPHS = -1

values_dict = {}

def choose_graph_dataset():
    name = input('Choose dataset node prediction [molhiv, molpcba, ppa, code]: ')
    name = 'ogbg-' + name
    ogb = ogbg.GraphPropPredDataset(name=name, root='/home/ctubert/tfg/gitprojects/gnn_analysis/analysis/datasets')
  
    return ogb

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

def get_biggest_CC(G, undirected):
    if undirected:
        CC = [G.subgraph(c).copy() for c in sorted(nx.algorithms.components.connected_components(G), key=len, reverse=True)]
    else:
        CC = [G.subgraph(c).copy() for c in sorted(nx.algorithms.components.strongly_connected_components(G), key=len, reverse=True)]

    num_CC = len(CC)
    cc = CC[0]
    num_nodes = cc.number_of_nodes()
    num_edges = cc.number_of_edges()

    values_dict['Num CC'] = num_CC
    values_dict['BCC num nodes'] = num_nodes
    values_dict['BCC num edges'] = num_edges

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

    values_dict['Num nodes'] = num_nodes
    values_dict['Num edges'] = num_edges
    values_dict['Average degree'] = avg_degree
    values_dict['Average clustering'] = avg_clustering
    values_dict['Density'] = density

def CC_processing(cc, undirected):
    print('Check directed cc', nx.is_directed(cc))
    num_nodes = cc.number_of_nodes()
    num_edges = cc.number_of_edges()

    avg_path_length = nx.average_shortest_path_length(cc)
    diameter = nx.diameter(cc)
    radius = nx.radius(cc)

    values_dict['BCC num nodes'] = num_nodes
    values_dict['BCC num edges'] = num_edges
    values_dict['BCC average path length'] = avg_path_length
    values_dict['BCC diameter'] = diameter
    values_dict['BCC radius'] = radius

def analysis(ogb):
    f = open('results_OGBG.csv', 'w', newline='')
    w = csv.writer(f)
    write_header = True

    num_graphs = len(ogb)

    for i in range(NUM_GRAPHS):
        graph_id = rand.randint(0, num_graphs)
        values_dict['Dataset name'] = ogb.name
        values_dict['Num graphs'] = num_graphs
        values_dict['Graph ID'] = graph_id

        ogb_graph = ogb[graph_id]

        nodes = list(range(0, ogb_graph[0]['num_nodes']))
        edge_index = ogb_graph[0]['edge_index']

        nodes_tensor = torch.LongTensor([x for x in nodes])
        edges_tensor = torch.LongTensor([x for x in edge_index])
        edges, _ = utils.subgraph(nodes_tensor, edges_tensor, num_nodes=len(nodes))

        G, undirected = get_nx_graph(nodes, edges)

        values_dict['Directed'] = (not undirected)

        time_ini = time.time()

        graph_processing(G, undirected)
        cc = get_biggest_CC(G, undirected)
        CC_processing(cc, undirected)

        time_end = time.time() - time_ini

        values_dict['Execution time'] = time_end

        if write_header:
            w.writerow(values_dict.keys())
            write_header = False

        print(i)
        w.writerow(values_dict.values())

    f.close()

def main():
    ogb = choose_graph_dataset()
    global NUM_GRAPHS
    NUM_GRAPHS = int(input('Choose NUM_GRAPHS: '))
    analysis(ogb)

if __name__ == '__main__':
  main()