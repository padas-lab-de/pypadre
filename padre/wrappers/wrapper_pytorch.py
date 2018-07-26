from collections import OrderedDict
import torch
import copy

# TODO: Add LR scheduler policy to code
# TODO: Implement Vision Layers
# TODO: Implement Dataparallel layers
# TODO: Implement utilities
# TODO: Implement modules and containers
# TODO: Implement parameter mapping file for layers to validate the parameter dictionary
# TODO: The above mapping file should contain the compulsory as well as optional parameters
# TODO: Batch processing to be implemented
# TODO: Implement rest of layers
# Types of layers Convolutional, Pooling, Padding, Non-Linear Activations,
# Normalization, Recurrent, Linear, Dropout, Sparse

class WrapperPytorch:

    model = None

    def __init__(self, params=None):
        """
        The initialization function for the pyTorch wrapper

        :param params: The parameters for creating the whole network

        """
        print('Initialize')
        if params is None:
            return

        self.params = copy.deepcopy(params)
        #self.lr = params.get('lr', 0.001)
        self.steps = params.get('steps', 1000)
        self.batch_size = params.get('batch_size', 1)

        architecture = params.get('architecture', None)
        layer_order = params.get('layer_order', None)
        shape = self.create_network_shape(architecture=architecture, layer_order=layer_order)
        self.model = self.create_model(shape)

        loss = params.get('loss', dict())
        loss_name = loss.get('name', 'MSELoss')
        loss_params = loss.get('params', None)
        self.loss = self.create_loss(loss_name, loss_params)

        optimizer = params.get('optimizer', dict())
        optimizer_type = optimizer.get('type', None)
        optimizer_params = optimizer.get('params', None)
        self.optimizer = self.create_optimizer(optimizer_type=optimizer_type,
                                               params=optimizer_params)

    def fit(self, x, y):
        """
        This function runs the training part of the experiment

        :param x: Training feature vectors
        :param y: Labels corresponding to the feature vectors

        :return: None
        """
        import numpy as np
        print('fit')

        # The output is always a 2 Dimensional matrix and y is reshaped for the shapes to be compatible
        if y.ndim == 1:
            y = np.reshape(y, newshape=(y.shape[0], 1))

        x = torch.autograd.Variable(torch.from_numpy(x), requires_grad=False)
        y = torch.autograd.Variable(torch.from_numpy(y), requires_grad=False)
        self.model = self.model.double()

        # Run the model for the steps specified in the parameters
        for step in range(self.steps):
            y_pred = self.model(x)
            loss = self.loss(y_pred, y)
            print(step, loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def predict(self, x):
        """
        This function tests the model created during training

        :param x: Input feature vectors

        :return: Predicted results
        """
        x = torch.autograd.Variable(torch.from_numpy(x), requires_grad=False)
        self.model.eval()
        with torch.no_grad():
            output =  self.model(x)

        return  output

    def create_model(self, shape):

        model = torch.nn.Sequential(*shape)
        return model

    def create_optimizer(self, optimizer_type=None, params=dict()):
        """
        This function implements an optimizer for learning
        Reference: https://pytorch.org/docs/stable/optim.html

        :param optimizer_type: The type of the optimizer
        :param params: Parameters of the optimizer

        :return: Optimizer object
        """

        lr = params.get('lr', 0.001)
        optimizer_type = str(optimizer_type).upper()

        optimizer = None

        # Create an object of the optimizer specified by the user.
        # Required parameters are given by the user within the params dictionary.
        # Missing parameters are substituted with default values obtained from the pytorch documentation.
        # Default optimizer is the Adam optimizer and SGD is selected if no match is found.

        if optimizer_type == 'ADADELTA':
            rho = params.get('rho', 0.9)
            eps = params.get('eps', 0.000001)
            weight_decay = params.get('weight_decay', 0)
            optimizer = torch.optim.Adadelta(self.model.parameters(), lr=lr, rho=rho, eps=eps, weight_decay=weight_decay)

        elif optimizer_type == 'ADAGRAD':
            lr_decay = params.get('lr_decay', 0)
            weight_decay = params.get('weight_decay', 0)
            optimizer = torch.optim.Adagrad(self.model.parameters(), lr=lr, lr_decay=lr_decay, weight_decay=weight_decay)

        elif optimizer_type == 'ADAM':
            betas = params.get('betas', (0.9, 0.999))
            eps = params.get('eps', 0.00000001)
            weight_decay = params.get('weight_decay', 0)
            amsgrad = params.get('amsgrad', False)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)

        elif optimizer_type == 'SPARSEADAM':
            betas = params.get('betas', (0.9, 0.999))
            eps = params.get('eps', 0.00000001)
            optimizer = torch.optim.SparseAdam(self.model.parameters(), lr=lr, betas=betas, eps=eps)

        elif optimizer_type == 'ADAMAX':
            betas = params.get('betas', (0.9, 0.999))
            eps = params.get('eps', 0.00000001)
            weight_decay = params.get('weight_decay', 0)
            optimizer = torch.optim.SparseAdam(self.model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        elif optimizer_type == 'ASGD':
            lambd = params.get('lambd', 0.0001)
            alpha = params.get('alpha', 0.75)
            t0 = params.get('t0', 1000000.0)
            weight_decay = params.get('weight_decay', 0)
            optimizer = torch.optim.ASGD(self.model.parameters(), lr=lr, lambd=lambd, alpha=alpha, t0=t0, weight_decay=weight_decay)

        elif optimizer_type == 'LBFGS':
            max_iter = params.get('max_iter', 20)
            max_eval = params.get('max_eval', None)
            tolerance_grad = params.get('tolerance_grad', 0.00001)
            tolerance_change = params.get('tolerance_change', 0.000000001)
            history_size = params.get('history_size', 100)
            line_search_fn = params.get('line_search_fn', None)
            optimizer = torch.optim.LBFGS(self.model.parameters(), lr=lr, max_iter=max_iter,
                                          max_eval=max_eval, tolerance_grad=tolerance_grad,
                                          tolerance_change=tolerance_change, history_size=history_size,
                                          line_search_fn=line_search_fn)

        elif optimizer_type == 'RMSPROP':
            alpha = params.get('alpha', 0.75)
            eps = params.get('eps', 0.00000001)
            weight_decay = params.get('weight_decay', 0)
            momentum = params.get('momentum', 0)
            centered = params.get('centered', False)
            optimizer = torch.optim.RMSprop(self.model.parameters(), lr=lr, alpha=alpha,
                                            eps=eps, weight_decay=weight_decay,
                                            momentum=momentum, centered=centered)

        elif optimizer_type == 'RPROP':
            etas = tuple(params.get('etas', (0.5, 1.2)))
            step_sizes = tuple(params.get('step_sizes', (0.000006, 50)))
            optimizer = torch.optim.Rprop(self.model.parameters(), lr=lr, etas=etas, step_sizes=step_sizes)

        elif optimizer_type == 'SGD':
            momentum = params.get('momentum', 0.9)
            dampening = params.get('dampening', 0)
            weight_decay = params.get('weight_decay', 0)
            nesterov = params.get('Nesterov', False)
            optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, dampening=dampening,
                                        weight_decay=weight_decay, nesterov=nesterov)

        else:
            momentum = params.get('momentum', 0.9)
            lr = params.get('lr', 0.1)
            optimizer = torch.optim.SGD(self.model.parameters(), lr)

        return optimizer

    def create_loss(self, name='MSELOSS', params=dict()):
        """
        The function returns an object of the required loss function

        :param name: Name of the loss funNonection
        :param params: The parameters of the loss function

        :return: Object of the loss function
        """

        loss = None



        name = str(name).upper()

        if name == 'L1LOSS':
            size_average = params.get('size_average', True)
            reduce = params.get('reduce', True)
            loss = torch.nn.L1Loss(size_average=size_average, reduce=reduce)

        elif name == 'MSELOSS':
            size_average = params.get('size_average', True)
            reduce = params.get('reduce', True)
            loss = torch.nn.MSELoss(size_average=size_average, reduce=reduce)

        elif name == 'CROSSENTROPYLOSS':
            weight = params.get('weight', None)
            size_average = params.get('size_average', True)
            ignore_index = params.get('ignore_index', -100)
            reduce = params.get('reduce', True)
            loss = torch.nn.CrossEntropyLoss(weight=weight, size_average=size_average,
                                             ignore_index=ignore_index, reduce=reduce)

        elif name == 'NLLLoss':
            weight = params.get('weight', None)
            size_average = params.get('size_average', True)
            ignore_index = params.get('ignore_index', -100)
            reduce = params.get('reduce', True)
            loss = torch.nn.NLLLoss(weight=weight, size_average=size_average, ignore_index=ignore_index, reduce=reduce)

        elif name == 'POISSONNLLLoss':
            log_input = params.get('log_input', True)
            full = params.get('full', False)
            size_average = params.get('size_average', True)
            eps = params.get('eps', 0.00000001)
            reduce = params.get('reduce', True)
            loss = torch.nn.PoissonNLLLoss(log_input=log_input, full=full, size_average=size_average,
                                           eps=eps, reduce=reduce)

        elif name == 'KLDIVLOSS':
            size_average = params.get('size_average', True)
            reduce = params.get('reduce', True)
            loss = torch.nn.KLDivLoss(size_average=size_average, reduce=reduce)

        elif name == 'BCELOSS':
            weight = params.get('weight', None)
            size_average = params.get('size_average', True)
            reduce = params.get('reduce', True)
            loss = torch.nn.BCELoss(weight=weight, size_average=size_average, reduce=reduce)

        elif name == 'BCEWITHLOGITSLOSS':
            weight = params.get('weight', None)
            size_average = params.get('size_average', True)
            reduce = params.get('reduce', True)
            loss = torch.nn.BCEWithLogitsLoss(weight=weight, size_average=size_average, reduce=reduce)

        elif name == 'MARGINRANKINGLOSS':
            margin = params.get('margin', 0)
            size_average = params.get('size_average', True)
            reduce = params.get('reduce', True)
            loss = torch.nn.MarginRankingLoss(margin=margin, size_average=size_average, reduce=reduce)

        elif name == 'HINGEEMBEDDINGLOSS':
            margin = params.get('margin', 1.0)
            size_average = params.get('size_average', True)
            reduce = params.get('reduce', True)
            loss = torch.nn.HingeEmbeddingLoss(margin=margin, size_average=size_average, reduce=reduce)

        elif name == 'MULTILABELMARGINLOSS':
            size_average = params.get('size_average', True)
            reduce = params.get('reduce', True)
            loss = torch.nn.MultiLabelMarginLoss(size_average=size_average, reduce=reduce)

        elif name == 'SMOOTHL1LOSS':
            size_average = params.get('size_average', True)
            reduce = params.get('reduce', True)
            loss = torch.nn.SmoothL1Loss(size_average=size_average, reduce=reduce)

        elif name == 'SOFTMARGINLOSS':
            size_average = params.get('size_average', True)
            reduce = params.get('reduce', True)
            loss = torch.nn.SoftMarginLoss(size_average=size_average, reduce=reduce)

        elif name == 'MULTILABELSOFTMARGINLOSS':
            weight = params.get('weight', None)
            size_average = params.get('size_average', True)
            reduce = params.get('reduce', True)
            loss = torch.nn.MultiLabelSoftMarginLoss(weight=weight, size_average=size_average, reduce=reduce)

        elif name == 'COSINEEMBEDDINGLOSS':
            margin = params.get('margin', 1.0)
            size_average = params.get('size_average', True)
            reduce = params.get('reduce', True)
            loss = torch.nn.CosineEmbeddingLoss(margin=margin, size_average=size_average, reduce=reduce)

        elif name == 'MULTIMARGINLOSS':
            p = params.get('p', 1)
            margin = params.get('margin', 1.0)
            weight = params.get('weight', None)
            size_average = params.get('size_average', True)
            reduce = params.get('reduce', True)
            loss = torch.nn.MultiMarginLoss(p=p, margin=margin, weight=weight, size_average=size_average, reduce=reduce)

        elif name == 'TRIPLETMARGINLOSS':
            margin = params.get('margin', 1.0)
            p = params.get('p', 2)
            eps = params.get('eps', 0.000001)
            swap = params.get('swap', False)
            size_average = params.get('size_average', True)
            reduce = params.get('reduce', True)
            loss = torch.nn.TripletMarginLoss(margin=margin, p=p, eps=eps, swap=swap,
                                              size_average=size_average, reduce=reduce)

        else:
            loss = torch.nn.MSELoss(size_average=False)

        return loss

    def create_network_shape(self, architecture=None, layer_order=None):
        """
        This function creates the network from the architecture specified in the config file

        :param architecture: The shape of the network
        :param layer_order: The ordering of the layers

        :return: The network object if the function succeeded else None
        """

        if architecture is None or layer_order is None:
            return None

        layers = []

        for layer in layer_order:

            layer = architecture.get(layer, None)
            if layer is None:
                return None

            type = layer.get('type')
            params = layer.get('params')

            layer_obj = self.create_layer_object(type, params)

            if layer_obj is not None:
                layers.append(layer_obj)

            else:
                layers = None
                break

        return layers

    def create_layer_object(self, layer_type = None, layer_params = None):
        """
        The function creates a layer object from the type and the params

        :param layer_type:
        :param layer_params:

        :return: A layer object
        """

        if layer_type is None:
            return None

        layer_type = str(layer_type).upper()

        if layer_params is None:
            layer_params = dict()

        layer_obj = None

        if layer_type == 'RELU':
            layer_obj = torch.nn.ReLU(**layer_params)

        elif layer_type == 'LINEAR':
            layer_obj = torch.nn.Linear(**layer_params)

        else:
            layer_obj = None

        if layer_obj is not None:
            return copy.deepcopy(layer_obj)

        else:
            return None










