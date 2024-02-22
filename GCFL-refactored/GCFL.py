import os
import argparse
import random
import torch
from pathlib import Path
import copy

import src.data_process as data_process
from src.train_func import *


def process_fedavg(clients: list, server: object) -> None:
    '''
    Entrance of running the FedAvg algorithm.

    :param clients: list of Client objects
    :param server: Server object
    '''
    print("\nDone setting up FedAvg devices.")
    print("Running FedAvg ...")

    outfile = os.path.join(outpath, f'accuracy_fedavg_GC.csv')

    frame = run_GC_fedavg(clients, server, args.num_rounds, args.local_epoch, samp=None)
    frame.to_csv(outfile)
    print(f"Wrote to file: {outfile}")


def process_gcfl(clients: list, server: object) -> None:
    '''
    Entrance of running the GCFL algorithm.

    :param clients: list of Client objects
    :param server: Server object
    '''
    print("\nDone setting up GCFL devices.")
    print("Running GCFL ...")

    outfile = os.path.join(outpath, f'accuracy_gcfl_GC.csv')

    frame = run_GC_gcfl(clients, server, args.num_rounds, args.local_epoch, EPS_1, EPS_2)
    frame.to_csv(outfile)
    print(f"Wrote to file: {outfile}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu',
                        help='CPU / GPU device.')
    parser.add_argument('--num_repeat', type=int, default=5,
                        help='number of repeating rounds to simulate;')
    parser.add_argument('--num_rounds', type=int, default=200,
                        help='number of rounds to simulate;')
    parser.add_argument('--local_epoch', type=int, default=1,
                        help='number of local epochs;')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate for inner solver;')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--nlayer', type=int, default=3,
                        help='Number of GINconv layers')
    parser.add_argument('--hidden', type=int, default=64,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for node classification.')
    parser.add_argument('--seed', help='seed for randomness;',
                        type=int, default=123)

    parser.add_argument('--datapath', type=str, default='./data',
                        help='The input path of data.')
    parser.add_argument('--outbase', type=str, default='./outputs',
                        help='The base path for outputting.')
    parser.add_argument('--repeat', help='index of repeating;',
                        type=int, default=None)
    parser.add_argument('--data_group', help='specify the group of datasets',
                        type=str, default='small')

    parser.add_argument('--convert_x', help='whether to convert original node features to one-hot degree features',
                        type=bool, default=False)
    parser.add_argument('--num_clients', help='number of clients',
                        type=int, default=10)
    parser.add_argument('--overlap', help='whether clients have overlapped data',
                        type=bool, default=False)
    parser.add_argument('--standardize', help='whether to standardize the distance matrix',
                        type=bool, default=False)
    parser.add_argument('--seq_length', help='the length of the gradient norm sequence',
                        type=int, default=5)
    parser.add_argument('--epsilon1', help='the threshold epsilon1 for GCFL',
                        type=float, default=0.05)
    parser.add_argument('--epsilon2', help='the threshold epsilon2 for GCFL',
                        type=float, default=0.1)

    try:
        args = parser.parse_args()
    except IOError as msg:
        parser.error(str(msg))

    print(args)
    seed_dataSplit = 123

    #################### set seeds and devices ####################
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    EPS_1 = args.epsilon1
    EPS_2 = args.epsilon2

    outbase = os.path.join(args.outbase, f'seqLen{args.seq_length}')
    outpath = os.path.join(outbase, f"oneDS-nonOverlap")
    outpath = os.path.join(outpath, f'{args.data_group}-{args.num_clients}clients', f'eps_{EPS_1}_{EPS_2}')
    Path(outpath).mkdir(parents=True, exist_ok=True)
    print(f"Output Path: {outpath}")

    #################### distributed one dataset to multiple clients ####################
    """ using original features """
    print("Preparing data (original features) ...")
    Path(os.path.join(outpath, 'repeats')).mkdir(parents=True, exist_ok=True)

    splitedData, df_stats = data_process.load_single_dataset(
        args.datapath, args.data_group, num_client=args.num_clients, batch_size=args.batch_size,
        convert_x=args.convert_x, seed=seed_dataSplit, overlap=args.overlap
    )
    print("Data prepared.")

    #################### save statistics of data on clients ####################
    outf = os.path.join(outpath, "repeats", f'{args.repeat}_stats_trainData.csv')
    df_stats.to_csv(outf)
    print(f"Wrote to {outf}")

    init_clients, _ = data_process.setup_clients(splitedData, args)
    init_server = data_process.setup_server(args)

    print("\nDone setting up devices.")

    #################### run FedAvg ####################
    # process_fedavg(clients=copy.deepcopy(init_clients), server=copy.deepcopy(init_server))
    
    #################### run GCFL ####################
    process_gcfl(clients=copy.deepcopy(init_clients), server=copy.deepcopy(init_server))

