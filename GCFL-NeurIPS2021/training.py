import pandas as pd
import numpy as np


def run_selftrain_GC(clients: list,
                     server: object,
                     local_epoch: int) -> dict:
    '''
    Run the training and testing process of self-training algorithm.
    It only trains the model locally, and does not perform weights aggregation.

    Args:
        clients: list of clients
        server: server object
        local_epoch: number of local epochs

    Returns:
        allAccs: dictionary of test accuracies
    '''
    # all clients are initialized with the same weights
    for client in clients:
        client.download_from_server(server)

    all_accs = {}
    for client in clients:
        client.local_train(local_epoch)

        _, acc = client.evaluate()
        all_accs[client.name] = [client.train_stats['trainingAccs'][-1], client.train_stats['valAccs'][-1], acc]
        print("  > {} done.".format(client.name))

    return all_accs


def run_fedavg(clients: list,
               server: object,
               COMMUNICATION_ROUNDS: int,
               local_epoch: int,
               samp: str = None,
               frac: float = 1.0) -> pd.DataFrame:
    '''
    Run the training and testing process of FedAvg algorithm.
    It trains the model locally, aggregates the weights to the server, 
    and downloads the global model within each communication round.

    Args:
        clients: list of clients
        server: server object
        COMMUNICATION_ROUNDS: number of communication rounds
        local_epoch: number of local epochs
        samp: sampling method
        frac: fraction of clients to sample

    Returns:
        frame: pandas dataframe with test accuracies
    '''

    for client in clients:
        client.download_from_server(server) # download the global model
    
    if samp is None:
        frac = 1.0

    # Overall training architecture: 
    # whole training => { communication rounds, communication rounds, ..., communication rounds }
    # communication rounds => { local training (#epochs) -> aggregation -> download }
    #                                |
    #                           training_stats
    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")   # print the current round every 50 rounds

        if c_round == 1:
            selected_clients = clients
        else:
            selected_clients = server.randomSample_clients(clients, frac)
            # if samp = None, frac=1.0, then all clients are selected

        for client in selected_clients:             # only get weights of graphconv layers
            client.local_train(local_epoch)         # train the local model

        server.aggregate_weights(selected_clients)  # aggregate the weights of selected clients
        for client in selected_clients:
            client.download_from_server(server)     # re-download the global server

    frame = pd.DataFrame()
    for client in clients:
        _, acc = client.evaluate()                  # Final evaluation
        frame.loc[client.name, 'test_acc'] = acc

    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: yellow' if v else '' for v in is_max]

    fs = frame.style.apply(highlight_max).data
    print(fs)
    return frame


def run_fedprox(clients: list,
                server: object,
                COMMUNICATION_ROUNDS: int, 
                local_epoch: int,
                mu: float,
                samp: str = None,
                frac: float = 1.0) -> pd.DataFrame:
    '''
    Run the training and testing process of FedProx algorithm.
    It trains the model locally, aggregates the weights to the server,
    and downloads the global model within each communication round.

    Args:
        clients: list of clients
        server: server object
        COMMUNICATION_ROUNDS: number of communication rounds
        local_epoch: number of local epochs
        mu: proximal term
        samp: sampling method
        frac: fraction of clients to sample

    Returns:
        frame: pandas dataframe with test accuracies
    '''
    for client in clients:
        client.download_from_server(server)

    if samp is None:
        frac = 1.0

    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")

        if c_round == 1:
            selected_clients = clients
        else:
            selected_clients = server.randomSample_clients(clients, frac)

        for client in selected_clients:
            client.local_train_prox(local_epoch, mu)    # Different from FedAvg

        server.aggregate_weights(selected_clients)
        for client in selected_clients:
            client.download_from_server(server)

            # cache the aggregated weights for next round
            client.cache_weights()

    frame = pd.DataFrame()
    for client in clients:
        _, acc = client.evaluate()
        frame.loc[client.name, 'test_acc'] = acc

    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: yellow' if v else '' for v in is_max]

    fs = frame.style.apply(highlight_max).data
    print(fs)
    return frame


def run_gcfl(clients: list, 
             server: object,
             COMMUNICATION_ROUNDS: int,
             local_epoch : int,
             EPS_1: float,
             EPS_2: float) -> pd.DataFrame:
    '''
    Run the GCFL algorithm.

    Args:
        clients: list of clients
        server: server object
        COMMUNICATION_ROUNDS: number of communication rounds
        local_epoch: number of local epochs
        EPS_1: threshold for mean update norm
        EPS_2: threshold for max update norm

    Returns:
        frame: pandas dataframe with test accuracies
    '''

    cluster_indices = [np.arange(len(clients)).astype("int")]   # cluster_indices: [[0, 1, ...]]
    client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]   # initially there is only one cluster

    ############### COMMUNICATION ROUNDS ###############
    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")
        if c_round == 1:
            for client in clients:
                client.download_from_server(server)

        participating_clients = server.randomSample_clients(clients, frac=1.0)
        for client in participating_clients:
            client.compute_weight_update(local_epoch)   # local training
            client.reset()                              # reset the gradients (discard the final gradients)

        similarities = server.compute_pairwise_similarities(clients)

        cluster_indices_new = []
        for idc in cluster_indices:  # cluster-wise checking 
            max_norm = server.compute_max_update_norm([clients[i] for i in idc])    # DELTA_MAX
            mean_norm = server.compute_mean_update_norm([clients[i] for i in idc])  # DELTA_MEAN
            if mean_norm < EPS_1 and max_norm > EPS_2 and len(idc) > 2 and c_round > 20:    # stopping condition
                server.cache_model(idc, clients[idc[0]].W, acc_clients)
                c1, c2 = server.min_cut(similarities[idc][:, idc], idc)
                cluster_indices_new += [c1, c2]   # split the cluster into two
            else:
                cluster_indices_new += [idc]      # keep the same cluster

        cluster_indices = cluster_indices_new
        client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]  # initial: [[0, 1, ...]]

        server.aggregate_clusterwise(client_clusters)   # aggregate the weights of the clients in each cluster

        acc_clients = [client.evaluate()[1] for client in clients]  # get the test accuracy of each client
    ############### END OF COMMUNICATION ROUNDS ###############
        
    for idc in cluster_indices:
        server.cache_model(idc, clients[idc[0]].W, acc_clients) # cache the first client's weights in each cluster
    # cluster-wise model

    results = np.zeros([len(clients), len(server.model_cache)])
    for i, (idcs, W, accs) in enumerate(server.model_cache):
        results[idcs, i] = np.array(accs)

    frame = pd.DataFrame(results, columns=["FL Model"] + ["Model {}".format(i)
                                                          for i in range(results.shape[1] - 1)],
                         index=["{}".format(clients[i].name) for i in range(results.shape[0])])
    frame = pd.DataFrame(frame.max(axis=1))
    frame.columns = ['test_acc']
    print(frame)

    return frame


def run_gcflplus(clients, server, COMMUNICATION_ROUNDS, local_epoch, EPS_1, EPS_2, seq_length, standardize):
    cluster_indices = [np.arange(len(clients)).astype("int")]
    client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]

    seqs_grads = {c.id:[] for c in clients}
    for client in clients:
        client.download_from_server(server)

    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")
        if c_round == 1:
            for client in clients:
                client.download_from_server(server)

        participating_clients = server.randomSample_clients(clients, frac=1.0)

        for client in participating_clients:
            client.compute_weight_update(local_epoch)
            client.reset()

            seqs_grads[client.id].append(client.convGradsNorm)

        cluster_indices_new = []
        for idc in cluster_indices:
            max_norm = server.compute_max_update_norm([clients[i] for i in idc])
            mean_norm = server.compute_mean_update_norm([clients[i] for i in idc])
            if mean_norm < EPS_1 and max_norm > EPS_2 and len(idc) > 2 and c_round > 20 \
                    and all(len(value) >= seq_length for value in seqs_grads.values()):

                server.cache_model(idc, clients[idc[0]].W, acc_clients)

                tmp = [seqs_grads[id][-seq_length:] for id in idc]
                dtw_distances = server.compute_pairwise_distances(tmp, standardize)
                c1, c2 = server.min_cut(np.max(dtw_distances)-dtw_distances, idc)
                cluster_indices_new += [c1, c2]

                seqs_grads = {c.id: [] for c in clients}
            else:
                cluster_indices_new += [idc]

        cluster_indices = cluster_indices_new
        client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]

        server.aggregate_clusterwise(client_clusters)

        acc_clients = [client.evaluate()[1] for client in clients]

    for idc in cluster_indices:
        server.cache_model(idc, clients[idc[0]].W, acc_clients)

    results = np.zeros([len(clients), len(server.model_cache)])
    for i, (idcs, W, accs) in enumerate(server.model_cache):
        results[idcs, i] = np.array(accs)

    frame = pd.DataFrame(results, columns=["FL Model"] + ["Model {}".format(i)
                                                          for i in range(results.shape[1] - 1)],
                         index=["{}".format(clients[i].name) for i in range(results.shape[0])])

    frame = pd.DataFrame(frame.max(axis=1))
    frame.columns = ['test_acc']
    print(frame)

    return frame


def run_gcflplus_dWs(clients, server, COMMUNICATION_ROUNDS, local_epoch, EPS_1, EPS_2, seq_length, standardize):
    cluster_indices = [np.arange(len(clients)).astype("int")]
    client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]

    seqs_grads = {c.id:[] for c in clients}
    for client in clients:
        client.download_from_server(server)

    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")
        if c_round == 1:
            for client in clients:
                client.download_from_server(server)

        participating_clients = server.randomSample_clients(clients, frac=1.0)

        for client in participating_clients:
            client.compute_weight_update(local_epoch)
            client.reset()

            seqs_grads[client.id].append(client.convDWsNorm)

        cluster_indices_new = []
        for idc in cluster_indices:
            max_norm = server.compute_max_update_norm([clients[i] for i in idc])
            mean_norm = server.compute_mean_update_norm([clients[i] for i in idc])
            if mean_norm < EPS_1 and max_norm > EPS_2 and len(idc) > 2 and c_round > 20 \
                    and all(len(value) >= seq_length for value in seqs_grads.values()):

                server.cache_model(idc, clients[idc[0]].W, acc_clients)

                tmp = [seqs_grads[id][-seq_length:] for id in idc]
                dtw_distances = server.compute_pairwise_distances(tmp, standardize)
                c1, c2 = server.min_cut(np.max(dtw_distances)-dtw_distances, idc)
                cluster_indices_new += [c1, c2]

                seqs_grads = {c.id: [] for c in clients}
            else:
                cluster_indices_new += [idc]

        cluster_indices = cluster_indices_new
        client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]

        server.aggregate_clusterwise(client_clusters)

        acc_clients = [client.evaluate()[1] for client in clients]

    for idc in cluster_indices:
        server.cache_model(idc, clients[idc[0]].W, acc_clients)

    results = np.zeros([len(clients), len(server.model_cache)])
    for i, (idcs, W, accs) in enumerate(server.model_cache):
        results[idcs, i] = np.array(accs)

    frame = pd.DataFrame(results, columns=["FL Model"] + ["Model {}".format(i)
                                                          for i in range(results.shape[1] - 1)],
                         index=["{}".format(clients[i].name) for i in range(results.shape[0])])
    frame = pd.DataFrame(frame.max(axis=1))
    frame.columns = ['test_acc']
    print(frame)

    return frame
