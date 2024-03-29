import torch
import numpy as np
import random
import networkx as nx
from dtaidistance import dtw


class ServerGC():
    '''
    This is a server class for federated graph classification which is responsible for aggregating model parameters from different clients, 
    updating the central model, and then broadcasting the updated model parameters back to the trainers.

    Parameters
    ----------
    model: torch.nn.Module
        The model to be trained.
    device: torch.device
        The device to run the model on.

    Attributes
    ----------
    model: serverGIN
        The central GIN model to be trained.
    W: dict
        Dictionary containing the model parameters.
    model_cache: list
        List of tuples, where each tuple contains the model parameters and the accuracies of the clients.
    '''
    def __init__(self, model: torch.nn.Module, device: torch.device) -> None:
        self.model = model.to(device)
        self.W = {key: value for key, value in self.model.named_parameters()}
        self.model_cache = []

    ########### Public functions ###########
    def random_sample_clients(self, all_clients: list, frac: float) -> list:
        '''
        Randomly sample a fraction of clients.

        Parameters
        ----------
        all_clients: list
            list of Client objects
        frac: float
            fraction of clients to be sampled

        Returns
        -------
        (sampled_clients): list
            list of Client objects
        '''
        return random.sample(all_clients, int(len(all_clients) * frac))

    def aggregate_weights(self, selected_clients: list) -> None:
        '''
        Perform weighted aggregation among selected clients. The weights are the number of training samples.

        Parameters
        ----------
        selected_clients: list
            list of Client objects
        '''
        total_size = 0
        for client in selected_clients:
            total_size += client.train_size
    
        for k in self.W.keys():
            # pass train_size, and weighted aggregate
            accumulate = torch.stack([torch.mul(client.W[k].data, client.train_size) for client in selected_clients])
            self.W[k].data = torch.div(torch.sum(accumulate, dim=0), total_size).clone()

    def compute_pairwise_similarities(self, clients: list) -> np.ndarray:
        '''
        This function computes the pairwise cosine similarities between the gradients of the clients.

        Parameters
        ----------
        clients: list
            list of Client objects

        Returns
        -------
        np.ndarray
            2D np.ndarray of shape len(clients) * len(clients), which contains the pairwise cosine similarities
        '''
        client_dWs = []
        for client in clients:
            dW = {}
            for k in self.W.keys():
                dW[k] = client.dW[k]
            client_dWs.append(dW)

        return self.__pairwise_angles(client_dWs)

    def compute_pairwise_distances(
            self, 
            seqs: list, 
            standardize: bool = False
    ) -> np.ndarray:
        '''
        This function computes the pairwise distances between the gradient norm sequences of the clients.

        Parameters
        ----------
        seqs: list
            list of 1D np.ndarray, where each 1D np.ndarray contains the gradient norm sequence of a client
        standardize: bool
            whether to standardize the distance matrix

        Returns
        -------
        distances: np.ndarray
            2D np.ndarray of shape len(seqs) * len(seqs), which contains the pairwise distances
        '''
        if standardize:
            # standardize to only focus on the trends
            seqs = np.array(seqs)
            seqs = seqs / seqs.std(axis=1).reshape(-1, 1)
            distances = dtw.distance_matrix(seqs)
        else:
            distances = dtw.distance_matrix(seqs)
        return distances

    def min_cut(self, similarity: np.ndarray, idc: list) -> tuple:
        '''
        This function computes the minimum cut of the graph defined by the pairwise cosine similarities.

        Parameters
        ----------
        similarity: np.ndarray
            2D np.ndarray of shape len(clients) * len(clients), which contains the pairwise cosine similarities
        idc: list
            list of client indices

        Returns
        -------
        (c1, c2): tuple
            tuple of two lists, where each list contains the indices of the clients in a cluster
        '''
        g = nx.Graph()
        for i in range(len(similarity)):
            for j in range(len(similarity)):
                g.add_edge(i, j, weight=similarity[i][j])
        _, partition = nx.stoer_wagner(g)   # using Stoer-Wagner algorithm to find the minimum cut
        c1 = np.array([idc[x] for x in partition[0]])
        c2 = np.array([idc[x] for x in partition[1]])
        return c1, c2

    def aggregate_clusterwise(self, client_clusters: list) -> None:
        '''
        Perform weighted aggregation among the clients in each cluster. 
        The weights are the number of training samples.

        Parameters
        ----------
        client_clusters: list
            list of lists, where each list contains the Client objects in a cluster
        '''
        for cluster in client_clusters:     # cluster is a list of Client objects
            targs, sours = [], []
            total_size = 0
            for client in cluster:
                W = {}
                dW = {}
                for k in self.W.keys():
                    W[k] = client.W[k]
                    dW[k] = client.dW[k]
                targs.append(W)
                sours.append((dW, client.train_size))
                total_size += client.train_size
            # pass train_size, and weighted aggregate
            self.__reduce_add_average(targets=targs, sources=sours, total_size=total_size)

    def compute_max_update_norm(self, cluster: list) -> float:
        '''
        Compute the maximum update norm (i.e., dW) among the clients in the cluster.
        This function is used to determine whether the cluster is ready to be split.

        Parameters
        ----------
        cluster: list
            list of Client objects
        '''
        max_dW = -np.inf
        for client in cluster:
            dW = {}
            for k in self.W.keys():
                dW[k] = client.dW[k]
            curr_dW = torch.norm(self.__flatten(dW)).item()
            max_dW = max(max_dW, curr_dW)

        return max_dW
    
    def compute_mean_update_norm(self, cluster: list) -> float:
        '''
        Compute the mean update norm (i.e., dW) among the clients in the cluster.
        This function is used to determine whether the cluster is ready to be split.

        Parameters
        ----------
        cluster: list
            list of Client objects
        '''
        cluster_dWs = []
        for client in cluster:
            dW = {}
            for k in self.W.keys():
                # dW[k] = client.dW[k]
                dW[k] = client.dW[k] * client.train_size / sum([c.train_size for c in cluster])
            cluster_dWs.append(self.__flatten(dW))

        return torch.norm(torch.mean(torch.stack(cluster_dWs), dim=0)).item()

    def cache_model(
            self, 
            idcs: list,
            params: dict,
            accuracies: list
    ) -> None:
        '''
        Cache the model parameters and accuracies of the clients.

        Parameters
        ----------
        idcs: list
            list of client indices
        params: dict
            dictionary containing the model parameters of the clients
        accuracies: list
            list of accuracies of the clients
        '''
        self.model_cache += [(idcs,
                              {name: params[name].data.clone() for name in params},
                              [accuracies[i] for i in idcs])]
        
    ########### Private functions ###########
    def __pairwise_angles(self, sources: list) -> np.ndarray:
        '''
        Compute the pairwise cosine similarities between the gradients of the clients into a 2D matrix.
        
        Parameters
        ----------
        sources: list
            list of dictionaries, where each dictionary contains the gradients of a client

        Returns
        -------
        np.ndarray
            2D np.ndarray of shape len(sources) * len(sources), which contains the pairwise cosine similarities
        '''
        angles = torch.zeros([len(sources), len(sources)])
        for i, source1 in enumerate(sources):
            for j, source2 in enumerate(sources):
                s1 = self.__flatten(source1)
                s2 = self.__flatten(source2)
                angles[i, j] = torch.true_divide(
                    torch.sum(s1 * s2), max(torch.norm(s1) * torch.norm(s2), 1e-12)) + 1

        return angles.numpy()

    def __flatten(self, source: dict) -> torch.Tensor:
        '''
        Flatten the gradients of a client into a 1D tensor.

        Parameters
        ----------
        source: dict
            dictionary containing the gradients of a client

        Returns
        -------
        (flattend_gradients): torch.Tensor
            1D tensor containing the flattened gradients
        '''
        return torch.cat([value.flatten() for value in source.values()])

    def __reduce_add_average(
            self, 
            targets: list, 
            sources: list, 
            total_size: int
    ) -> None:
        '''
        Perform weighted aggregation from the sources to the targets. The weights are the number of training samples.

        Parameters
        ----------
        targets: list
            list of dictionaries, where each dictionary contains the model parameters of a client
        sources: list
            list of tuples, where each tuple contains the gradients and the number of training samples of a client
        total_size: int
            total number of training samples
        '''
        for target in targets:
            for name in target:
                weighted_stack = torch.stack([torch.mul(source[0][name].data, source[1]) for source in sources])
                tmp = torch.div(torch.sum(weighted_stack, dim=0), total_size).clone()
                target[name].data += tmp