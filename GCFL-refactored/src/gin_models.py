import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool


class serverGIN(torch.nn.Module):
    '''
    This class defines the server model for the GIN model.

    Parameters
    ----------
    nlayer: int
        The number of layers in the GIN model.
    nhid: int
        The number of hidden units in the GIN model.

    Attributes
    ----------
    graph_convs: torch.nn.ModuleList
        The list of graph convolutional layers.
    nn1: torch.nn.Sequential
        The first neural network layer.
    nnk: torch.nn.Sequential
        The k-th neural network layer.
    '''
    def __init__(self, nlayer: int, nhid: int) -> None:
        super(serverGIN, self).__init__()
        self.graph_convs = torch.nn.ModuleList()
        self.nn1 = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU(),
                                       torch.nn.Linear(nhid, nhid))
        self.graph_convs.append(GINConv(self.nn1))
        
        for _ in range(nlayer - 1):
            self.nnk = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU(),
                                           torch.nn.Linear(nhid, nhid))
            self.graph_convs.append(GINConv(self.nnk))


class GIN(torch.nn.Module):
    '''
    A Graph Isomorphism Network (GIN) model implementation which creates a GIN with specified
    numbers of features, hidden units, classes, layers, and dropout.

    Parameters
    ----------
    nfeat: int
        The number of input features.
    nhid: int
        The number of hidden features in each layer of the GIN model.
    nclass: int
        The number of output classes.
    nlayer: int
        The number of layers.
    dropout: float
        The dropout rate.

    Attributes
    ----------
    num_layers: int
        The number of layers in the GIN model.
    dropout: float
        The dropout rate.
    pre: torch.nn.Sequential
        The pre-neural network layer.
    graph_convs: torch.nn.ModuleList
        The list of graph convolutional layers.
    nn1: torch.nn.Sequential
        The first neural network layer.
    nnk: torch.nn.Sequential
        The k-th neural network layer.
    post: torch.nn.Sequential
        The post-neural network layer.
    '''
    def __init__(
        self, 
        nfeat: int, 
        nhid: int, 
        nclass: int, 
        nlayer: int, 
        dropout: float
    ) -> None:
        super(GIN, self).__init__()
        self.num_layers = nlayer
        self.dropout = dropout
        self.pre = torch.nn.Sequential(torch.nn.Linear(nfeat, nhid))
        self.graph_convs = torch.nn.ModuleList()
        self.nn1 = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU(), torch.nn.Linear(nhid, nhid))
        self.graph_convs.append(GINConv(self.nn1))

        for _ in range(nlayer - 1):
            self.nnk = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU(), torch.nn.Linear(nhid, nhid))
            self.graph_convs.append(GINConv(self.nnk))

        self.post = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU())
        self.readout = torch.nn.Sequential(torch.nn.Linear(nhid, nclass))

    def forward(self, data: torch_geometric.data.Data) -> torch.Tensor:
        '''
        Forward pass of the GIN model, which takes in the input graph data and returns the 
        model's prediction.

        Parameters
        ----------
        data: torch_geometric.data.Data
            The input graph data.

        Returns
        -------
        torch.Tensor
            The prediction of the model.
        '''
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.pre(x)
        for i in range(len(self.graph_convs)):
            x = self.graph_convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        x = global_add_pool(x, batch)
        x = self.post(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.readout(x)
        x = F.log_softmax(x, dim=1)
        return x

    def loss(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        '''
        Compute the loss of the model.

        Parameters
        ----------
        pred: torch.Tensor
            The prediction of the model.
        label: torch.Tensor
            The label of the input data.

        Return
        ------
        torch.Tensor
            The nll loss of the model.
        '''
        return F.nll_loss(pred, label)