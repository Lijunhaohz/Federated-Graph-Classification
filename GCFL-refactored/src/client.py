import torch


class Client_GC():
    def __init__(self, model, client_id, client_name, train_size, dataLoader, optimizer, args) -> None:
        '''
        Initialize the client model.
        '''
        self.model = model.to(args.device)
        self.id = client_id
        self.name = client_name
        self.train_size = train_size
        self.dataLoader = dataLoader
        self.optimizer = optimizer
        self.args = args

        self.W = {key: value for key, value in self.model.named_parameters()}
        self.dW = {key: torch.zeros_like(value) for key, value in self.model.named_parameters()}
        self.W_old = {key: value.data.clone() for key, value in self.model.named_parameters()}

        self.gconvNames = None

        self.train_stats = ([0], [0], [0], [0])
        self.weightsNorm = 0.
        self.gradsNorm = 0.
        self.convGradsNorm = 0.
        self.convWeightsNorm = 0.
        self.convDWsNorm = 0.


    def download_from_server(self, server: object) -> None:
        '''
        Download the global model weights from the server.

        Args:
        - server: Server object
        '''
        self.gconvNames = server.W.keys()   # gconv layers
        for k in server.W:
            self.W[k].data = server.W[k].data.clone()


    def cache_weights(self) -> dict:
        '''
        Cache the weights of the model.

        Returns:
        - dict: the cached weights
        '''
        for name in self.W.keys():
            self.W_old[name].data = self.W[name].data.clone()


    def reset(self) -> None:
        '''
        Reset the weights of the model to the cached weights.
        The implementation is copying the cached weights (W_old) to the model weights (W).

        '''
        copy(target=self.W, source=self.W_old, keys=self.gconvNames)


    def set_stats_norms(self, train_stats: dict, is_gcfl: bool = False) -> None:
        '''
        Set the norms of the weights and gradients of the model.

        Args:
        - train_stats: dict, the training statistics
        - is_gcfl: bool, whether the model is trained with GCFL
        '''
        self.train_stats = train_stats

        self.weightsNorm = torch.norm(flatten(self.W)).item()

        weights_conv = {key: self.W[key] for key in self.gconvNames}
        self.convWeightsNorm = torch.norm(flatten(weights_conv)).item()

        grads = {key: value.grad for key, value in self.W.items()}
        self.gradsNorm = torch.norm(flatten(grads)).item()

        grads_conv = {key: self.W[key].grad for key in self.gconvNames}
        self.convGradsNorm = torch.norm(flatten(grads_conv)).item()

        if is_gcfl:
            dWs_conv = {key: self.dW[key] for key in self.gconvNames}
            self.convDWsNorm = torch.norm(flatten(dWs_conv)).item()


    def local_train(self, local_epoch: int) -> None:
        """ 
        Train the model locally for self-train & FedAvg 

        Args:
        - local_epoch: int, the number of local epochs
        """
        train_stats = train_gc(model=self.model, 
                               dataloaders=self.dataLoader, 
                               optimizer=self.optimizer, 
                               local_epoch=local_epoch, 
                               device=self.args.device)

        self.set_stats_norms(train_stats)


    def local_train_prox(self, local_epoch: int, mu: float) -> None:
        '''
        Train the model locally for FedProx with the proximal term.

        Args:
        - local_epoch: int, the number of local epochs
        - mu: float, the proximal term coefficient
        '''
        train_stats = train_gc_prox(model=self.model, 
                                    dataloaders=self.dataLoader, 
                                    optimizer=self.optimizer, 
                                    local_epoch=local_epoch, 
                                    device=self.args.device,
                                    gconvNames=self.gconvNames, 
                                    Ws=self.W, 
                                    mu=mu, 
                                    Wt=self.W_old)

        self.set_stats_norms(train_stats)


    def compute_weight_update(self, local_epoch: int) -> None:
        """ 
        Train the model locally for GCFL.

        Args:
        - local_epoch: int, the number of local epochs
        """

        copy(target=self.W_old, source=self.W, keys=self.gconvNames)

        train_stats = train_gc(model=self.model, 
                               dataloaders=self.dataLoader, 
                               optimizer=self.optimizer, 
                               local_epoch=local_epoch, 
                               device=self.args.device)

        subtract_(target=self.dW, minuend=self.W, subtrahend=self.W_old)    # dW = W - W_old

        self.set_stats_norms(train_stats, is_gcfl=True)


    def evaluate(self) -> tuple:
        '''
        Final evaluation of the model on the test dataset.

        Returns:
        - tuple(float, float): the average loss and accuracy
        '''
        return eval_gc(self.model, self.dataLoader['test'], self.args.device)


    def evaluate_prox(self, mu):
        return eval_gc_prox(self.model, self.dataLoader['test'], self.args.device, self.gconvNames, mu, self.W_old)


def copy(target: dict, source: dict, keys: list) -> None:
    '''
    Copy the source weights to the target weights.

    Args:
    - target: dict, the target weights
    - source: dict, the source weights
    - keys: list, the names of the layers
    '''
    for name in keys:
        target[name].data = source[name].data.clone()


def subtract_(target: dict, minuend: dict, subtrahend: dict) -> None:
    '''
    Subtract the subtrahend from the minuend and store the result in the target.

    Args:
    - target: dict, the target weights
    - minuend: dict, the minuend weights
    - subtrahend: dict, the subtrahend weights
    '''
    for name in target:
        target[name].data = minuend[name].data.clone() - subtrahend[name].data.clone()


def flatten(w: dict) -> torch.tensor:
    '''
    Flatten the gradients of a client into a 1D tensor.

    Args:
    - w: dict, the gradients of a client

    Returns:
    - torch.tensor: the flattened gradients
    '''
    return torch.cat([v.flatten() for v in w.values()])


def calc_gradsNorm(gconvNames: list, Ws: dict) -> float:
    '''
    Calculate the norm of the gradients of the gconv layers.

    Args:
    - gconvNames: list, the names of the gconv layers
    - Ws: dict, the weights of the model

    Returns:
    - float: the norm of the gradients of the gconv layers
    '''
    grads_conv = {k: Ws[k].grad for k in gconvNames}
    convGradsNorm = torch.norm(flatten(grads_conv)).item()
    return convGradsNorm


def train_gc(model: object, 
             dataloaders: dict, 
             optimizer: object, 
             local_epoch: int, 
             device: str) -> dict:
    '''
    Train the model on the local dataset.

    Args:
    - model: object, the model to be trained
    - dataloaders: dict, the dataloaders for training, validation, and testing
    - optimizer: object, the optimizer for training
    - local_epoch: int, the number of local epochs
    - device: str, the device to run the training

    Returns:
    - dict: the training statistics
    '''

    losses_train, accs_train, losses_val, accs_val, losses_test, accs_test = [], [], [], [], [], []
    train_loader, val_loader, test_loader = dataloaders['train'], dataloaders['val'], dataloaders['test']
    for _ in range(local_epoch):
        model.train()
        loss_train, acc_train, num_graphs = 0., 0., 0

        for _, batch in enumerate(train_loader):
            batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            label = batch.y
            loss = model.loss(pred, label)
            loss.backward()
            optimizer.step()

            loss_train += loss.item() * batch.num_graphs
            acc_train += pred.max(dim=1)[1].eq(label).sum().item()
            num_graphs += batch.num_graphs

        loss_train /= num_graphs    # get the average loss per graph
        acc_train /= num_graphs  # get the average average per graph

        loss_val, acc_val = eval_gc(model, val_loader, device)
        loss_test, acc_test = eval_gc(model, test_loader, device)

        losses_train.append(loss_train)
        accs_train.append(acc_train)
        losses_val.append(loss_val)
        accs_val.append(acc_val)
        losses_test.append(loss_test)
        accs_test.append(acc_test)

    # record the losses and accuracies for each epoch
    return {'trainingLosses': losses_train, 'trainingAccs': accs_train, 
            'valLosses': losses_val, 'valAccs': accs_val,
            'testLosses': losses_test, 'testAccs': accs_test}


def eval_gc(model: object,
            test_loader: object,
            device: str) -> tuple:
    '''
    Validate and test the model on the local dataset.

    Args:
    - model: object, the model to be evaluated
    - test_loader: object, the dataloader for testing
    - device: str, the device to run the evaluation

    Returns:
    - tuple(float, float): the average loss and accuracy
    '''

    model.eval()
    total_loss, total_acc, num_graphs = 0., 0., 0

    for batch in test_loader:
        batch.to(device)
        with torch.no_grad():
            pred = model(batch)
            label = batch.y
            loss = model.loss(pred, label)

        total_loss += loss.item() * batch.num_graphs
        total_acc += pred.max(dim=1)[1].eq(label).sum().item()
        num_graphs += batch.num_graphs

    return total_loss / num_graphs, total_acc / num_graphs


def _prox_term(model: object, gconvNames: list, Wt: dict) -> torch.tensor:
    '''
    Compute the proximal term.

    Args:
    - model: object, the model to be trained
    - gconvNames: list, the names of the gconv layers
    - Wt: dict, the target weights

    Returns:
    - torch.tensor: the proximal term
    '''
    prox = torch.tensor(0., requires_grad=True)
    for name, param in model.named_parameters():
        if name in gconvNames:      # only add the prox term for sharing layers (gConv)
            prox = prox + torch.norm(param - Wt[name]).pow(2)   # force the weights to be close to the old weights
    return prox


def train_gc_prox(model: object,
                  dataloaders: dict,
                  optimizer: object,
                  local_epoch: int,
                  device: str,
                  gconvNames: list,
                  Ws: dict, 
                  mu: float, 
                  Wt: dict) -> dict:
    '''
    Train the model on the local dataset with the proximal term.

    Args:
    - model: object, the model to be trained
    - dataloaders: dict, the dataloaders for training, validation, and testing
    - optimizer: object, the optimizer for training
    - local_epoch: int, the number of local epochs
    - device: str, the device to run the training
    - gconvNames: list, the names of the gconv layers
    - Ws: dict, the weights of the model
    - mu: float, the proximal term
    - Wt: dict, the target weights (W_old)

    Returns:
    - dict: the training statistics
    '''
    losses_train, accs_train, losses_val, accs_val, losses_test, accs_test = [], [], [], [], [], []
    convGradsNorm = []
    train_loader, val_loader, test_loader = dataloaders['train'], dataloaders['val'], dataloaders['test']
    for _ in range(local_epoch):
        model.train()
        total_loss = 0.
        ngraphs = 0
        acc_sum = 0

        for _, batch in enumerate(train_loader):
            batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            label = batch.y
            acc_sum += pred.max(dim=1)[1].eq(label).sum().item()
            loss = model.loss(pred, label) + mu / 2. * _prox_term(model, gconvNames, Wt)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            ngraphs += batch.num_graphs
        total_loss /= ngraphs
        acc = acc_sum / ngraphs

        loss_val, acc_val = eval_gc(model, val_loader, device)
        loss_test, acc_test = eval_gc(model, test_loader, device)

        losses_train.append(total_loss)
        accs_train.append(acc)
        losses_val.append(loss_val)
        accs_val.append(acc_val)
        losses_test.append(loss_test)
        accs_test.append(acc_test)

        convGradsNorm.append(calc_gradsNorm(gconvNames, Ws))

    return {'trainingLosses': losses_train, 'trainingAccs': accs_train, 
            'valLosses': losses_val, 'valAccs': accs_val,
            'testLosses': losses_test, 'testAccs': accs_test, 
            'convGradsNorm': convGradsNorm}


def eval_gc_prox(model, test_loader, device, gconvNames, mu, Wt):
    model.eval()

    total_loss = 0.
    acc_sum = 0.
    ngraphs = 0
    for batch in test_loader:
        batch.to(device)
        with torch.no_grad():
            pred = model(batch)
            label = batch.y
            loss = model.loss(pred, label) + mu / 2. * _prox_term(model, gconvNames, Wt)
        total_loss += loss.item() * batch.num_graphs
        acc_sum += pred.max(dim=1)[1].eq(label).sum().item()
        ngraphs += batch.num_graphs

    return total_loss / ngraphs, acc_sum / ngraphs