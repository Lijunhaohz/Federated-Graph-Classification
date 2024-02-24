import os
import argparse
import random
from pathlib import Path
import copy
import yaml

import numpy as np
import torch

from .data_process_gc import *
from .train_func import *
from .utils_gc import *


def GC_Train(config: dict):
    '''
    Entrance of the training process for graph classification.

    Parameters
    ----------
    model: str
        The model to run.
    '''
    # transfer the config to argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    for key, value in config.items():
        setattr(args, key, value)

    print(args)

    #################### set seeds and devices ####################
    seed_split_data = 42    # seed for splitting data must be fixed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    #################### set output directory ####################
    # outdir_base = os.path.join(args.outbase, f'seqLen{args.seq_length}')
    outdir_base = args.outbase + '/' + f'{args.model}'
    outdir = os.path.join(outdir_base, f"oneDS-nonOverlap")
    if args.model in ['SelfTrain']:
        outdir = os.path.join(outdir, f'{args.data_group}')
    elif args.model in ['FedAvg', 'FedProx']:
        outdir = os.path.join(outdir, f'{args.data_group}-{args.num_clients}clients')
    elif args.model in ['GCFL']:
        outdir = os.path.join(outdir, f'{args.data_group}-{args.num_clients}clients', 
                              f'eps_{args.epsilon1}_{args.epsilon2}')
    elif args.model in ['GCFL+', 'GCFL+dWs']:
        outdir = os.path.join(outdir, f'{args.data_group}-{args.num_clients}clients', 
                              f'eps_{args.epsilon1}_{args.epsilon2}', f'seqLen{args.seq_length}')

    Path(outdir).mkdir(parents=True, exist_ok=True)
    print(f"Output Path: {outdir}")

    #################### distributed one dataset to multiple clients ####################
    """ using original features """
    print("Preparing data (original features) ...")

    splited_data, df_stats = load_single_dataset(
        args.datapath, args.data_group, num_client=args.num_clients, batch_size=args.batch_size,
        convert_x=args.convert_x, seed=seed_split_data, overlap=args.overlap
    )
    print("Data prepared.")

    #################### save statistics of data on clients ####################
    outdir_stats = os.path.join(outdir, f'stats_train_data.csv')
    df_stats.to_csv(outdir_stats)
    print(f"The statistics of the data are written to {outdir_stats}")

    #################### setup devices ####################
    if args.model not in ['SelfTrain']:
        init_clients, _ = setup_clients(splited_data, args)
        init_server = setup_server(args)
        clients = copy.deepcopy(init_clients)
        server = copy.deepcopy(init_server)

    print("\nDone setting up devices.")

    ################ choose the model to run ################
    print(f"Running {args.model} ...")
    if args.model == 'SelfTrain':
        output = run_GC_selftrain(
            clients=clients, server=server, local_epoch=args.local_epoch)

    elif args.model == 'FedAvg':
        output = run_GC_fedavg(
            clients=clients, server=server, communication_rounds=args.num_rounds, 
            local_epoch=args.local_epoch, samp=None)
        
    elif args.model == 'FedProx':
        output = run_GC_fedprox(
            clients=clients, server=server, communication_rounds=args.num_rounds, 
            local_epoch=args.local_epoch, mu=args.mu, samp=None)
        
    elif args.model == 'GCFL':
        output = run_GC_gcfl(
            clients=clients, server=server, communication_rounds=args.num_rounds, 
            local_epoch=args.local_epoch, EPS_1=args.epsilon1, EPS_2=args.epsilon2)
        
    elif args.model == 'GCFL+':
        output = run_GC_gcfl_plus(
            clients=clients, server=server, communication_rounds=args.num_rounds, 
            local_epoch=args.local_epoch, EPS_1=args.epsilon1, EPS_2=args.epsilon2,
            seq_length=args.seq_length, standardize=args.standardize)
        
    elif args.model == 'GCFL+dWs':
        output = run_GC_gcfl_plus(
            clients=clients, server=server, communication_rounds=args.num_rounds, 
            local_epoch=args.local_epoch, EPS_1=args.epsilon1, EPS_2=args.epsilon2,
            seq_length=args.seq_length, standardize=args.standardize)
        
    else:
        raise ValueError(f"Unknown model: {args.model}")
        
    #################### save the output ####################
    outdir_result = os.path.join(outdir, f'accuracy_seed{args.seed}.csv')
    output.to_csv(outdir_result)
    print(f"The output has been written to file: {outdir_result}")


if __name__ == "__main__":
    GC_Train(model='GCFL')
