import torch


class Trainer_GC():
    '''
    A trainer class specified for graph classification tasks, which includes functionalities required
    for training GIN models on a subset of a distributed dataset, handling local training and testing, parameter updates, and feature aggregation.
    
    Parameters
    ----------
    model: object
        The model to be trained, which is based on the GIN model.
    client_id: int
        The ID of the client.
    client_name: str
        The name of the client.
    train_size: int
        The size of the training dataset.
    dataLoader: dict
        The dataloaders for training, validation, and testing.
    optimizer: object
        The optimizer for training.
    args: object
        The arguments for the training.

    Attributes
    ----------
    model: object
        The model to be trained, which is based on the GIN model.
    id: int
        The ID of the client.
    name: str
        The name of the client.
    train_size: int
        The size of the training dataset.
    dataloader: dict
        The dataloaders for training, validation, and testing.
    optimizer: object
        The optimizer for training.
    args: object
        The arguments for the training.
    W: dict
        The weights of the model.
    dW: dict
        The gradients of the model.
    W_old: dict
        The cached weights of the model.
    gconv_names: list
        The names of the gconv layers.
    train_stats: tuple
        The training statistics of the model.
    weights_norm: float
        The norm of the weights of the model.
    grads_norm: float
        The norm of the gradients of the model.
    conv_grads_norm: float
        The norm of the gradients of the gconv layers.
    conv_weights_Norm: float
        The norm of the weights of the gconv layers.
    conv_dWs_norm: float
        The norm of the gradients of the gconv layers.
    '''
    def __init__(
            self, 
            model: object, 
            client_id: int, 
            client_name: str, 
            train_size: int, 
            dataloader: dict, 
            optimizer: object, 
            args: object
    ) -> None:
        self.model = model.to(args.device)
        self.id = client_id
        self.name = client_name
        self.train_size = train_size
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.args = args

        self.W = {key: value for key, value in self.model.named_parameters()}
        self.dW = {key: torch.zeros_like(value) for key, value in self.model.named_parameters()}
        self.W_old = {key: value.data.clone() for key, value in self.model.named_parameters()}

        self.gconv_names = None

        self.train_stats = ([0], [0], [0], [0])
        self.weights_norm = 0.
        self.grads_norm = 0.
        self.conv_grads_norm = 0.
        self.conv_weights_norm = 0.
        self.conv_dWs_norm = 0.

    ########### Public functions ###########
    def update_params(self, server: object) -> None:
        '''
        Update the model parameters by downloading the global model weights from the server.

        Parameters
        ----------
        server: object
            The server object that contains the global model weights.
        '''
        self.gconvNames = server.W.keys()   # gconv layers
        for k in server.W:
            self.W[k].data = server.W[k].data.clone()

    def reset_params(self) -> None:
        '''
        Reset the weights of the model to the cached weights.
        The implementation is copying the cached weights (W_old) to the model weights (W).

        '''
        self.__copy_weights(target=self.W, source=self.W_old, keys=self.gconvNames)

    def cache_weights(self) -> dict:
        '''
        Cache the weights of the model.

        Returns
        -------
        (cached_weights): dict
            The cached weights of the model.
        '''
        for name in self.W.keys():
            self.W_old[name].data = self.W[name].data.clone()

    def set_stats_norms(self, train_stats: dict, is_gcfl: bool = False) -> None:
        '''
        Set the norms of the weights and gradients of the model, as well as the statistics of the training.

        Parameters
        ----------
        train_stats: dict
            The training statistics of the model.
        is_gcfl: bool, optional
            Whether the training is for GCFL. The default is False.
        '''
        self.train_stats = train_stats

        self.weightsNorm = torch.norm(self.__flatten(self.W)).item()

        weights_conv = {key: self.W[key] for key in self.gconvNames}
        self.convWeightsNorm = torch.norm(self.__flatten(weights_conv)).item()

        grads = {key: value.grad for key, value in self.W.items()}
        self.gradsNorm = torch.norm(self.__flatten(grads)).item()

        grads_conv = {key: self.W[key].grad for key in self.gconvNames}
        self.convGradsNorm = torch.norm(self.__flatten(grads_conv)).item()

        if is_gcfl: # special case for GCFL
            dWs_conv = {key: self.dW[key] for key in self.gconvNames}
            self.convDWsNorm = torch.norm(self.__flatten(dWs_conv)).item()

    def local_train(
            self, 
            local_epoch: int, 
            train_option: str = 'basic', 
            mu: float = 1
    ) -> None:
        """ 
        This function is a interface of the trainer class to train the model locally.
        It will call the train function specified for the training option, based on the args provided.

        Parameters
        ----------
        local_epoch: int
            The number of local epochs
        train_option: str, optional
            The training option. The possible values are 'basic', 'prox', and 'gcfl'. The default is 'basic'.
            'basic' - self-train and FedAvg
            'prox' - FedProx
            'gcfl' - GCFL, GCFL+ and GCFL+dWs
        mu: float, optional
            The proximal term. The default is 1.
        """
        assert train_option in ['basic', 'prox', 'gcfl'], "Invalid training option."

        if train_option == 'gcfl':
            self.__copy_weights(target=self.W_old, source=self.W, keys=self.gconvNames)

        if train_option in ['basic', 'prox']:
            train_stats = self.__train(
                model=self.model, 
                dataloaders=self.dataLoader, 
                optimizer=self.optimizer, 
                local_epoch=local_epoch, 
                device=self.args.device
            )
        elif train_option == 'gcfl':
            train_stats = self.__train(
                model=self.model, 
                dataloaders=self.dataLoader, 
                optimizer=self.optimizer, 
                local_epoch=local_epoch, 
                device=self.args.device,
                prox=True, 
                gconv_names=self.gconvNames, 
                Ws=self.W, 
                Wt=self.W_old, 
                mu=mu
            )

        if train_option == 'gcfl':
            self.__subtract_weights(target=self.dW, minuend=self.W, subtrahend=self.W_old)

        self.set_stats_norms(train_stats)

    def local_test(
            self, 
            test_option: str = 'basic',
            mu: float = 1
    ) -> tuple:
        '''
        Final test of the model on the test dataset based on the test option.

        Parameters
        ----------
        test_option: str, optional
            The test option. The possible values are 'basic' and 'prox'. The default is 'basic'.
            'basic' - self-train, FedAvg, GCFL, GCFL+ and GCFL+dWs
            'prox' - FedProx
        mu: float, optional
            The proximal term. The default is 1.
        '''
        assert test_option in ['basic', 'prox'], "Invalid test option."

        if test_option == 'basic':
            return self.__eval(model=self.model, test_loader=self.dataLoader['test'], device=self.args.device)
        elif test_option == 'prox':
            return self.__eval(model=self.model, test_loader=self.dataLoader['test'], device=self.args.device,
                                prox=True, gconv_names=self.gconvNames, mu=mu, Wt=self.W_old)
    
    ########### Private functions ###########
    def __train(
        self,
        model: object, 
        dataloaders: dict, 
        optimizer: object, 
        local_epoch: int, 
        device: str,
        prox: bool = False,
        gconv_names: list = None,
        Ws: dict = None,
        Wt: dict = None,
        mu: float = None
    ) -> dict:
        '''
        Train the model on the local dataset.

        Parameters
        ----------
        model: object
            The model to be trained
        dataloaders: dict
            The dataloaders for training, validation, and testing
        optimizer: object
            The optimizer for training
        local_epoch: int
            The number of local epochs
        device: str
            The device to run the training

        Returns
        -------
        (results): dict
            The training statistics
        '''
        if prox:
            assert ((gconv_names is not None) and (Ws is not None) 
                and (Wt is not None) and (mu is not None)), "Please provide the required arguments for the proximal term."

        losses_train, accs_train, losses_val, accs_val, losses_test, accs_test = [], [], [], [], [], []
        if prox:
            convGradsNorm = []
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
                loss += mu / 2. * self.__prox_term(model, gconv_names, Wt) if prox else 0.     # add the proximal term if required
                loss.backward()
                optimizer.step()
                loss_train += loss.item() * batch.num_graphs
                acc_train += pred.max(dim=1)[1].eq(label).sum().item()
                num_graphs += batch.num_graphs

            loss_train /= num_graphs    # get the average loss per graph
            acc_train /= num_graphs  # get the average average per graph

            loss_val, acc_val = self.__eval(model, val_loader, device)
            loss_test, acc_test = self.__eval(model, test_loader, device)

            losses_train.append(loss_train)
            accs_train.append(acc_train)
            losses_val.append(loss_val)
            accs_val.append(acc_val)
            losses_test.append(loss_test)
            accs_test.append(acc_test)

            if prox:
                convGradsNorm.append(self.__calc_grads_norm(gconv_names, Ws))

        # record the losses and accuracies for each epoch
        res_dict = {'trainingLosses': losses_train, 'trainingAccs': accs_train,
                    'valLosses': losses_val, 'valAccs': accs_val,
                    'testLosses': losses_test, 'testAccs': accs_test}
        if prox:
            res_dict['convGradsNorm'] = convGradsNorm

        return res_dict
    
    def __eval(
            self,
            model: object,
            test_loader: object,
            device: str,
            prox: bool = False,
            gconv_names: list = None,
            mu: float = None,
            Wt: dict = None
    ) -> tuple:
        '''
        Validate and test the model on the local dataset.

        Parameters
        ----------
        model: object
            The model to be tested
        test_loader: object
            The dataloader for testing
        device: str
            The device to run the testing
        prox: bool, optional
            Whether to add the proximal term. The default is False.
        gconv_names: list, optional
            The names of the gconv layers. The default is None.
        mu: float, optional
            The proximal term. The default is None.
        Wt: dict, optional
            The target weights. The default is None.

        Returns
        -------
        (test_loss, test_acc): tuple(float, float)
            The average loss and accuracy
        '''
        if prox:
            assert ((gconv_names is not None) and (mu is not None) 
                    and (Wt is not None)), "Please provide the required arguments for the proximal term."

        model.eval()
        total_loss, total_acc, num_graphs = 0., 0., 0

        for batch in test_loader:
            batch.to(device)
            with torch.no_grad():
                pred = model(batch)
                label = batch.y
                loss = model.loss(pred, label)
                loss += mu / 2. * self.__prox_term(model, gconv_names, Wt) if prox else 0.

            total_loss += loss.item() * batch.num_graphs
            total_acc += pred.max(dim=1)[1].eq(label).sum().item()
            num_graphs += batch.num_graphs

        return total_loss / num_graphs, total_acc / num_graphs

    def __prox_term(
            self, 
            model: object, 
            gconvNames: list, 
            Wt: dict
    ) -> torch.tensor:
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
    
    def __calc_grads_norm(
            self, 
            gconvNames: list, 
            Ws: dict
    ) -> float:
        '''
        Calculate the norm of the gradients of the gconv layers.

        Args:
        - gconvNames: list, the names of the gconv layers
        - Ws: dict, the weights of the model

        Returns:
        - float: the norm of the gradients of the gconv layers
        '''
        grads_conv = {k: Ws[k].grad for k in gconvNames}
        convGradsNorm = torch.norm(self.__flatten(grads_conv)).item()
        return convGradsNorm


    def __copy_weights(
            self, 
            target: dict, 
            source: dict, 
            keys: list
    ) -> None:
        '''
        Copy the source weights to the target weights.

        Args:
        - target: dict, the target weights
        - source: dict, the source weights
        - keys: list, the names of the layers
        '''
        for name in keys:
            target[name].data = source[name].data.clone()


    def __subtract_weights(
            self, 
            target: dict, 
            minuend: dict, 
            subtrahend: dict
    ) -> None:
        '''
        Subtract the subtrahend from the minuend and store the result in the target.

        Args:
        - target: dict, the target weights
        - minuend: dict, the minuend weights
        - subtrahend: dict, the subtrahend weights
        '''
        for name in target:
            target[name].data = minuend[name].data.clone() - subtrahend[name].data.clone()


    def __flatten(self, w: dict) -> torch.tensor:
        '''
        Flatten the gradients of a client into a 1D tensor.

        Args:
        - w: dict, the gradients of a client

        Returns:
        - torch.tensor: the flattened gradients
        '''
        return torch.cat([v.flatten() for v in w.values()])
