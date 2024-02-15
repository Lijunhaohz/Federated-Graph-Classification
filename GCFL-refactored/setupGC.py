import random
from random import choices
import argparse

import numpy as np
import pandas as pd
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.transforms import OneHotDegree

from models import GIN, serverGIN
from server import Server
from client import Client_GC
from utils import get_maxDegree, get_stats, split_data, get_numGraphLabels


def _randChunk(graphs: list,
               num_client: int=10, 
               overlap: bool=False, 
               seed: int=None) -> list:
    '''
    Randomly split graphs into chunks for each client.

    Args:
    - graphs: list, the list of graphs.
    - num_client: int, the number of clients.
    - overlap: bool, whether clients have overlapped data.

    Returns:
    - graphs_chunks: list, the list of chunks for each client.
    '''
    
    random.seed(seed)
    np.random.seed(seed)

    totalNum = len(graphs)
    minSize = min(50, int(totalNum / num_client))
    graphs_chunks = []
    if not overlap:
        for i in range(num_client):
            graphs_chunks.append(graphs[i * minSize : (i + 1) * minSize])
        for g in graphs[num_client * minSize:]:
            idx_chunk = np.random.randint(low=0, high=num_client, size=1)[0]
            graphs_chunks[idx_chunk].append(g)
    else:
        sizes = np.random.randint(low=50, high=150, size=num_client)
        for s in sizes:
            graphs_chunks.append(choices(graphs, k=s))
    return graphs_chunks


def prepareData_oneDS(datapath: str,
                      dataset: str='PROTEINS', 
                      num_client: int=10, 
                      batchSize: int=128, 
                      convert_x: bool=False, 
                      seed: int=None,
                      overlap: bool=False) -> (dict, pd.DataFrame):
    '''
    Prepare data for one dataset to multiple clients.

    Args:
    - datapath: str, the input path of data.
    - dataset: str, the name of dataset.
    - num_client: int, the number of clients.
    - batchSize: int, the batch size for node classification.
    - convert_x: bool, whether to convert node features to one-hot degree.
    - seed: int, seed for randomness.
    - overlap: bool, whether clients have overlapped data.

    Returns:
    - splitedData: dict, the data for each client.
    - df: pd.DataFrame, the statistics of data.
    '''
    
    if dataset == "COLLAB":
        tudataset = TUDataset(f"{datapath}/TUDataset", dataset, pre_transform=OneHotDegree(491, cat=False))
    elif dataset == "IMDB-BINARY":
        tudataset = TUDataset(f"{datapath}/TUDataset", dataset, pre_transform=OneHotDegree(135, cat=False))
    elif dataset == "IMDB-MULTI":
        tudataset = TUDataset(f"{datapath}/TUDataset", dataset, pre_transform=OneHotDegree(88, cat=False))
    else:
        tudataset = TUDataset(f"{datapath}/TUDataset", dataset)
        if convert_x:
            maxdegree = get_maxDegree(tudataset)
            tudataset = TUDataset(f"{datapath}/TUDataset", dataset, transform=OneHotDegree(maxdegree, cat=False))
    
    graphs = [x for x in tudataset]
    print("Dataset name: ", dataset, " Total number of graphs: ", len(graphs))

    ''' Split data into chunks for each client '''
    graphs_chunks = _randChunk(graphs=graphs, 
                               num_client=num_client, 
                               overlap=overlap, 
                               seed=seed)
    
    splitedData = {}
    stats_df = pd.DataFrame()
    num_node_features = graphs[0].num_node_features

    for idx, chunks in enumerate(graphs_chunks):
        ds = f'{idx}-{dataset}'    # client id

        '''Data split'''
        ds_whole = chunks
        ds_train, ds_val_test = split_data(ds_whole, train=0.8, test=0.2, shuffle=True, seed=seed)
        ds_val, ds_test = split_data(ds_val_test, train=0.5, test=0.5, shuffle=True, seed=seed)

        '''Generate data loader'''
        dataloader_train = DataLoader(ds_train, batch_size=batchSize, shuffle=True)
        dataloader_val = DataLoader(ds_val, batch_size=batchSize, shuffle=True)
        dataloader_test = DataLoader(ds_test, batch_size=batchSize, shuffle=True)
        num_graph_labels = get_numGraphLabels(ds_train)

        '''Combine data'''
        splitedData[ds] = ({'train': dataloader_train, 'val': dataloader_val, 'test': dataloader_test},
                           num_node_features, num_graph_labels, len(ds_train))
        
        '''Statistics'''
        stats_df = get_stats(df=stats_df, 
                             ds_client_name=ds, 
                             graphs_train=ds_train, 
                             graphs_val=ds_val, 
                             graphs_test=ds_test)

    return splitedData, stats_df


def prepareData_multiDS(datapath: str,
                        dataset_group: str='small', 
                        batchSize: int=32, 
                        convert_x: bool=False, 
                        seed: int=None):
    '''
    Prepare data for multiple datasets.

    Args:
    - datapath: str, the input path of data.
    - dataset_group: str, the name of dataset group.
    - batchSize: int, the batch size for node classification.
    - convert_x: bool, whether to convert node features to one-hot degree.
    - seed: int, seed for randomness.

    Returns:
    - splitedData: dict, the data for each client.
    - df: pd.DataFrame, the statistics of data.
    '''

    assert dataset_group in ['molecules', 'molecules_tiny', 'small', 'mix', "mix_tiny", "biochem", "biochem_tiny"]

    if dataset_group == 'molecules' or dataset_group == 'molecules_tiny':
        datasets = ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1"]
    if dataset_group == 'small':
        datasets = ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR",                   # small molecules
                    "ENZYMES", "DD", "PROTEINS"]                                # bioinformatics
    if dataset_group == 'mix' or dataset_group == 'mix_tiny':
        datasets = ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1",   # small molecules
                    "ENZYMES", "DD", "PROTEINS",                                # bioinformatics
                    "COLLAB", "IMDB-BINARY", "IMDB-MULTI"]                      # social networks
    if dataset_group == 'biochem' or dataset_group == 'biochem_tiny':
        datasets = ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1",  # small molecules
                    "ENZYMES", "DD", "PROTEINS"]                               # bioinformatics

    splitedData = {}
    df = pd.DataFrame()

    for dataset in datasets:
        if dataset == "COLLAB":
            tudataset = TUDataset(f"{datapath}/TUDataset", dataset, pre_transform=OneHotDegree(491, cat=False))
        elif dataset == "IMDB-BINARY":
            tudataset = TUDataset(f"{datapath}/TUDataset", dataset, pre_transform=OneHotDegree(135, cat=False))
        elif dataset == "IMDB-MULTI":
            tudataset = TUDataset(f"{datapath}/TUDataset", dataset, pre_transform=OneHotDegree(88, cat=False))
        else:
            tudataset = TUDataset(f"{datapath}/TUDataset", dataset)
            if convert_x:
                maxdegree = get_maxDegree(tudataset)
                tudataset = TUDataset(f"{datapath}/TUDataset", dataset, transform=OneHotDegree(maxdegree, cat=False))

        graphs = [x for x in tudataset]
        print("Dataset name: ", dataset, " Total number of graphs: ", len(graphs))

        '''Split data'''
        graphs_train, graphs_valtest = split_data(graphs, test=0.2, shuffle=True, seed=seed)
        graphs_val, graphs_test = split_data(graphs_valtest, train=0.5, test=0.5, shuffle=True, seed=seed)

        if dataset_group.endswith('tiny'):
            graphs, _ = split_data(graphs, train=150, shuffle=True, seed=seed)
            graphs_train, graphs_val_test = split_data(graphs, test=0.2, shuffle=True, seed=seed)
            graphs_val, graphs_test = split_data(graphs_val_test, train=0.5, test=0.5, shuffle=True, seed=seed)

        num_node_features = graphs[0].num_node_features
        num_graph_labels = get_numGraphLabels(graphs_train)

        '''Generate data loader'''
        dataloader_train = DataLoader(graphs_train, batch_size=batchSize, shuffle=True)
        dataloader_val = DataLoader(graphs_val, batch_size=batchSize, shuffle=True)
        dataloader_test = DataLoader(graphs_test, batch_size=batchSize, shuffle=True)

        '''Combine data'''
        splitedData[dataset] = ({'train': dataloader_train, 'val': dataloader_val, 'test': dataloader_test},
                             num_node_features, num_graph_labels, len(graphs_train))

        '''Statistics'''
        df = get_stats(df, dataset, graphs_train, graphs_val=graphs_val, graphs_test=graphs_test)

    return splitedData, df


def setup_clients(splitedData: dict, 
                  args: argparse.ArgumentParser=None) -> tuple:
    '''
    Setup clients

    Args:
    - splitedData: dict, the data for each client.
    - args: argparse.ArgumentParser, the input arguments.

    Returns:
    - clients: list, the list of clients.
    - idx_clients: dict, the index of clients.
    '''

    idx_clients = {}
    clients = []
    for idx, dataset_client_name in enumerate(splitedData.keys()):
        idx_clients[idx] = dataset_client_name
        '''acquire data'''
        dataloaders, num_node_features, num_graph_labels, train_size = splitedData[dataset_client_name]

        '''build GIN model'''
        cmodel_gc = GIN(nfeat=num_node_features, 
                        nhid=args.hidden, 
                        nclass=num_graph_labels, 
                        nlayer=args.nlayer, 
                        dropout=args.dropout)
       
        '''build optimizer'''
        optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, cmodel_gc.parameters()), 
                                     lr=args.lr, 
                                     weight_decay=args.weight_decay)

        '''build client'''
        client = Client_GC(model=cmodel_gc,                     # GIN model
                           client_id=idx,                       # client id
                           client_name=dataset_client_name,     # client name
                           train_size=train_size,               # training size
                           dataLoader=dataloaders,              # data loader
                           optimizer=optimizer,                 # optimizer
                           args=args)

        clients.append(client)

    return clients, idx_clients


def setup_server(args: argparse.ArgumentParser=None) -> Server:
    '''
    Setup server.

    Args:
    - args: argparse.ArgumentParser, the input arguments.

    Returns:
    - server: Server, the server.
    '''

    smodel = serverGIN(nlayer=args.nlayer, nhid=args.hidden)
    server = Server(smodel, args.device)
    return server


# def setup_devices(splitedData, 
#                   args=None) -> (list, Server, dict):
#     '''
#     Setup devices for clients and server.

#     Args:
#     - splitedData: dict, the data for each client.
#     - args: argparse.ArgumentParser, the input arguments.

#     Returns:
#     - clients: list, the list of clients.
#     - server: Server, the server.
#     - idx_clients: dict, the index of clients.
#     '''

#     idx_clients = {}
#     clients = []
#     for idx, dataset_client in enumerate(splitedData.keys()):
#         idx_clients[idx] = dataset_client
#         dataloaders, num_node_features, num_graph_labels, train_size = splitedData[dataset_client]
#         cmodel_gc = GIN(num_node_features, args.hidden, num_graph_labels, args.nlayer, args.dropout)
#         # optimizer = torch.optim.Adam(cmodel_gc.parameters(), lr=args.lr, weight_decay=args.weight_decay)
#         optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, cmodel_gc.parameters()), lr=args.lr, weight_decay=args.weight_decay)
#         clients.append(Client_GC(cmodel_gc, idx, dataset_client, train_size, dataloaders, optimizer, args))

#     smodel = serverGIN(nlayer=args.nlayer, nhid=args.hidden)
#     server = Server(smodel, args.device)
#     return clients, server, idx_clients