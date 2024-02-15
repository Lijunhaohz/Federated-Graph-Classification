import pandas as pd
import torch
from torch_geometric.utils import to_networkx, degree
import torch.nn.functional as F
from sklearn.model_selection import train_test_split


def convert_to_nodeDegreeFeatures(graphs):
    graph_infos = []
    maxdegree = 0
    for i, graph in enumerate(graphs):
        g = to_networkx(graph, to_undirected=True)
        gdegree = max(dict(g.degree).values())
        if gdegree > maxdegree:
            maxdegree = gdegree
        graph_infos.append((graph, g.degree, graph.num_nodes))    # (graph, node_degrees, num_nodes)

    new_graphs = []
    for i, tuple in enumerate(graph_infos):
        idx, x = tuple[0].edge_index[0], tuple[0].x
        deg = degree(idx, tuple[2], dtype=torch.long)
        deg = F.one_hot(deg, num_classes=maxdegree + 1).to(torch.float)

        new_graph = tuple[0].clone()
        new_graph.__setitem__('x', deg)
        new_graphs.append(new_graph)

    return new_graphs


def get_maxDegree(graphs) -> int:
    '''
    Get the maximum degree of the graphs in the dataset.

    Args:
    - graphs: list of graphs

    Returns:
    - int: maximum degree
    '''
    maxdegree = 0
    for i, graph in enumerate(graphs):
        g = to_networkx(graph, to_undirected=True)
        gdegree = max(dict(g.degree).values())
        if gdegree > maxdegree:
            maxdegree = gdegree

    return maxdegree


def use_node_attributes(graphs):
    num_node_attributes = graphs.num_node_attributes
    new_graphs = []
    for i, graph in enumerate(graphs):
        new_graph = graph.clone()
        new_graph.__setitem__('x', graph.x[:, :num_node_attributes])
        new_graphs.append(new_graph)
    return new_graphs


def split_data(graphs: list,
               train=None, 
               test=None, 
               shuffle: bool=True, 
               seed: int=None) -> (list, list):
    '''
    Split the dataset into training and test sets.

    Args:
    - graphs: list of graphs
    - train: float or int, if float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split. If int, represents the absolute number of train samples.
    - test: float or int, if float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. If int, represents the absolute number of test samples.
    - shuffle: bool, whether or not to shuffle the data before splitting.
    - seed: int, seed for the random number generator.

    Returns:
    - graphs_train: list of training graphs
    - graphs_test: list of testing graphs
    
    '''
    y = torch.cat([graph.y for graph in graphs])
    graphs_train, graphs_test = train_test_split(graphs, train_size=train, test_size=test, stratify=y, shuffle=shuffle, random_state=seed)
    return graphs_train, graphs_test


def get_numGraphLabels(dataset: list) -> int:
    '''
    Get the number of unique graph labels in the dataset.

    Args:
    - dataset: list of graphs

    Returns:
    - int: number of unique graph labels
    '''
    s = set()
    for g in dataset:
        s.add(g.y.item())
    return len(s)


def _get_avg_nodes_edges(graphs: list) -> (float, float):
    '''
    Calculate the average number of nodes and edges in the dataset.

    Args:
    - graphs: list of graphs

    Returns:
    - avgNodes: average number of nodes
    - avgEdges: average number of edges
    '''
    numNodes = 0.
    numEdges = 0.
    numGraphs = len(graphs)
    for g in graphs:
        numNodes += g.num_nodes
        numEdges += g.num_edges / 2.  # undirected

    avgNodes = numNodes / numGraphs
    avgEdges = numEdges / numGraphs
    return avgNodes, avgEdges


def get_stats(df: pd.DataFrame,
              ds_client_name: str,
              graphs_train: list,
              graphs_val: list=None, 
              graphs_test: list=None) -> pd.DataFrame:
    '''
    Calculate and store the statistics of the dataset.

    Args:
    - df: pd.DataFrame, the statistics of data.
    - ds_client_name: str, the name of the dataset accompanied by the client id.
    - graphs_train: list of training graphs.
    - graphs_val: list of validation graphs.
    - graphs_test: list of testing graphs.

    Returns:
    - df: pd.DataFrame, the updated statistics of data.
    '''
    
    df.loc[ds_client_name, "#graphs_train"] = len(graphs_train)
    avgNodes, avgEdges = _get_avg_nodes_edges(graphs_train)
    df.loc[ds_client_name, 'avgNodes_train'] = avgNodes
    df.loc[ds_client_name, 'avgEdges_train'] = avgEdges

    if graphs_val:
        df.loc[ds_client_name, '#graphs_val'] = len(graphs_val)
        avgNodes, avgEdges = _get_avg_nodes_edges(graphs_val)
        df.loc[ds_client_name, 'avgNodes_val'] = avgNodes
        df.loc[ds_client_name, 'avgEdges_val'] = avgEdges

    if graphs_test:
        df.loc[ds_client_name, '#graphs_test'] = len(graphs_test)
        avgNodes, avgEdges = _get_avg_nodes_edges(graphs_test)
        df.loc[ds_client_name, 'avgNodes_test'] = avgNodes
        df.loc[ds_client_name, 'avgEdges_test'] = avgEdges

    return df