from collections import OrderedDict
import torch
import copy
import numpy as np
import json
import importlib
import os
import random
from torch.nn import Module
# TODO: Add LR scheduler policy to code
# TODO: Implement Vision Layers
# TODO: Implement Dataparallel layers
# TODO: Implement utilities
# TODO: Implement modules and containers
# TODO: Batch processing to be implemented
# TODO: Implement Recurrent Layers


class Flatten(Module):
    """
    This class is used to flatten the output of a layer inorder to pass it to a fully connected layer
    """
    def forward(self, input):
        """
        Returns a reshaped array of the input
        :param input: A multidimensional array
        :return: A two dimensional array with the first dimension same
        """
        #print ('Flatten Input size:' + str(input.size()))
        #temp = input.view(input.size()[0], -1)
        #print('Flatten Output Size:' + str(temp.size()))
        #return temp
        return input.view(input.size()[0], -1)


class TestLayer(Module):
    """
    This is a debugging layer to pause the forward pass
    """
    def forward(self, input):
        print ('Test Input size:' + str(input.size()))
        return input


class WrapperPytorch:

    model = None

    layers_dict = None

    top_shape = 0

    probabilities = None

    flatten = False

    checkpoint = 0

    resume = False

    pre_trained_model_path = None

    def __init__(self, params=None):
        """
        The initialization function for the pyTorch wrapper

        :param params: The parameters for creating the whole network

        """
        if params is None:
            return

        with open('mappings_torch.json') as f:
            self.layers_dict = json.load(f)

        self.params = copy.deepcopy(params)
        self.steps = params.get('steps', 1000)
        self.checkpoint = params.get('checkpoint', self.steps)
        self.batch_size = params.get('batch_size', 1)
        self.resume = params.get('resume', False)
        self.pre_trained_model_path = params.get('model', None)
        self.model_prefix = params.get('model_prefix', "")

        architecture = params.get('architecture', None)
        layer_order = params.get('layer_order', None)
        shape = self.create_network_shape(architecture=architecture, layer_order=layer_order)

        # Failed network creation
        if shape is None:
            return

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

        self.probabilities = None

        training_samples_count = y.shape[0]


        # The output is always a 2 Dimensional matrix and y is reshaped for the shapes to be compatible
        if y.ndim == 1:
            if self.top_shape == 1:
                y = np.reshape(y, newshape=(y.shape[0], 1))

            else:
                # Do one hot encoding
                from numpy import array
                from numpy import argmax
                from sklearn.preprocessing import LabelEncoder
                from sklearn.preprocessing import OneHotEncoder

                label_encoder = LabelEncoder()
                integer_encoded = label_encoder.fit_transform(y)
                onehot_encoder = OneHotEncoder(sparse=False)
                integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
                y = onehot_encoder.fit_transform(integer_encoded)

        x = torch.autograd.Variable(torch.from_numpy(x), requires_grad=False)
        y = torch.autograd.Variable(torch.from_numpy(y), requires_grad=False)
        self.model = self.model.double()

        permutation = torch.randperm(x.size()[0])
        start_idx = 0

        if self.batch_size > x.size()[0]:
            batch_size = x.size()[0]

        else:
            batch_size = self.batch_size

        randomize = False

        # Run the model for the steps specified in the parameters
        step = 0

        if self.resume is True and os.path.isfile(self.pre_trained_model_path):
            state = torch.load(self.pre_trained_model_path)
            self.optimizer.load_state_dict(state['optimizer'])
            self.model.load_state_dict(state['model'])
            step = state['step']

        while step < self.steps:

            if randomize is True:
                permutation = torch.randperm(x.size()[0])

            indices = permutation[start_idx: start_idx + batch_size]

            if start_idx + batch_size > training_samples_count:
                indices = permutation[start_idx:training_samples_count]
                randomize = True
                indices = np.append(indices, permutation[0:start_idx + batch_size - training_samples_count])

            start_idx = (start_idx + batch_size) % training_samples_count
            x_mini_batch = x[indices]
            y_mini_batch = y[indices]

            y_pred = self.model(x_mini_batch)
            loss = self.loss(y_pred, y_mini_batch)
            print(step+1, loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if step % self.checkpoint == 0:
                prefix = self.model_prefix
                save_file_name = "_".join([prefix, "model", str(step)])
                state = dict()
                state['step'] = step
                state['model'] = self.model.state_dict()
                state['optimizer'] = self.optimizer.state_dict()
                torch.save(state, save_file_name)

            step = step + 1

        prefix = self.model_prefix
        save_file_name = "_".join([prefix, "model", str(step)])
        state = dict()
        state['step'] = step
        state['model'] = self.model.state_dict()
        state['optimizer'] = self.optimizer.state_dict()
        torch.save(state, save_file_name)

    def predict(self, x):
        """
        This function tests the model created during training

        :param x: Input feature vectors

        :return: Predicted results
        """
        test_samples_count = x.shape[0]
        start_idx = 0
        batch_size = self.batch_size

        x = torch.autograd.Variable(torch.from_numpy(x), requires_grad=False)
        self.model.eval()

        if self.top_shape > 1:
            output = np.zeros(shape=(test_samples_count, self.top_shape))

        else:
            output = np.zeros(shape=(test_samples_count,1))

        with torch.no_grad():
            while start_idx <= test_samples_count:
                indices = torch.LongTensor(list(range(start_idx, start_idx+batch_size)))
                end_idx = start_idx+batch_size
                if end_idx > test_samples_count:
                    indices = torch.LongTensor(list(range(start_idx, test_samples_count)))
                    end_idx = test_samples_count

                mini_batch_x = x[indices]
                mini_batch_output = self.model(mini_batch_x)
                output[start_idx:end_idx,:] = mini_batch_output
                start_idx = start_idx + batch_size


        if mini_batch_output.shape[1] > 1 and self.top_shape > 1:
            self.probabilities = output
            output = np.argmax(output, axis=1)

        else:
            output = np.reshape(output, newshape=(output.shape[0]))

        return output

    def predict_proba(self, x):

        if self.probabilities is None:
            probabilites = np.zeros(shape=(len(x), self.top_shape))

        else:
            probabilities = self.probabilities

        return probabilities


    def create_model(self, shape):

        model = torch.nn.Sequential(*shape)
        return model

    def create_optimizer(self, optimizer_type=None, params=None):
        """
        This function implements an optimizer for learning
        Reference: https://pytorch.org/docs/stable/optim.html

        :param optimizer_type: The type of the optimizer
        :param params: Parameters of the optimizer

        :return: Optimizer object
        """

        if params is None:
            params = dict()

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
            optimizer = torch.optim.Adadelta(self.model.parameters(), lr=lr, rho=rho,
                                             eps=eps, weight_decay=weight_decay)

        elif optimizer_type == 'ADAGRAD':
            lr_decay = params.get('lr_decay', 0)
            weight_decay = params.get('weight_decay', 0)
            optimizer = torch.optim.Adagrad(self.model.parameters(), lr=lr, lr_decay=lr_decay,
                                            weight_decay=weight_decay)

        elif optimizer_type == 'ADAM':
            betas = params.get('betas', (0.9, 0.999))
            eps = params.get('eps', 0.00000001)
            weight_decay = params.get('weight_decay', 0)
            amsgrad = params.get('amsgrad', False)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=betas, eps=eps,
                                         weight_decay=weight_decay, amsgrad=amsgrad)

        elif optimizer_type == 'SPARSEADAM':
            betas = params.get('betas', (0.9, 0.999))
            eps = params.get('eps', 0.00000001)
            optimizer = torch.optim.SparseAdam(self.model.parameters(), lr=lr, betas=betas, eps=eps)

        elif optimizer_type == 'ADAMAX':
            betas = params.get('betas', (0.9, 0.999))
            eps = params.get('eps', 0.00000001)
            weight_decay = params.get('weight_decay', 0)
            optimizer = torch.optim.Adamax(self.model.parameters(), lr=lr, betas=betas,
                                           eps=eps, weight_decay=weight_decay)

        elif optimizer_type == 'ASGD':
            lambd = params.get('lambd', 0.0001)
            alpha = params.get('alpha', 0.75)
            t0 = params.get('t0', 1000000.0)
            weight_decay = params.get('weight_decay', 0)
            optimizer = torch.optim.ASGD(self.model.parameters(), lr=lr, lambd=lambd, alpha=alpha,
                                         t0=t0, weight_decay=weight_decay)

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
            dampening = params.get('dampening', 0)
            weight_decay = params.get('weight_decay', 0)
            nesterov = params.get('Nesterov', False)
            optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, dampening=dampening,
                                        weight_decay=weight_decay, nesterov=nesterov)

        return optimizer

    def create_loss(self, name='MSELOSS', params=None):
        """
        The function returns an object of the required loss function

        :param name: Name of the loss funNonection
        :param params: The parameters of the loss function

        :return: Object of the loss function
        """

        if params is None:
            params = dict()

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

        for layer_name in layer_order:

            layer = architecture.get(layer_name, None)
            if layer is None:
                return None

            layer_type = layer.get('type')
            params = layer.get('params')

            layer_obj = self.create_layer_object(layer_type, params)

            if layer_obj is not None:
                if str(layer_type).upper() in ['LINEAR', 'BILINEAR']:
                    self.top_shape = params.get('out_features')
                    if self.flatten is False:
                        layers.append(Flatten())
                        self.flatten = True
                layers.append(layer_obj)

            else:
                layers = None
                break

        return layers

    def create_layer_object(self, layer_type=None, layer_params=None):
        """
        The function creates a layer object from the type and the params
        Reference: https://pytorch.org/docs/stable/nn.html

        :param layer_type:
        :param layer_params:

        :return: A layer object
        """

        if layer_type is None:
            return None

        layer_type = str(layer_type).upper()

        if layer_params is None:
            layer_params = dict()

        # Verify params for the module
        layer = self.layers_dict.get(layer_type, None)

        if layer is None:
            return None

        path = layer.get('path', None)

        if path is None:
            return None

        # Some layers have no parameters while some layers might have not defined any parameters.
        # The latter case is an error and the string is used to distinguish between both cases.
        params = layer.get('params', "PARAMSNOTDEFINED")
        if params == 'PARAMSNOTDEFINED':
            return None

        curr_params = dict()
        for param in params:
            param_value = layer_params.get(param, None)
            if param_value is None and params.get(param).get('optional') is False:
                curr_params = None
                break

            else:
                if param_value is not None:
                    curr_params[param] = param_value

        obj = None
        if curr_params is not None:
            split_idx = path.rfind('.')
            import_path = path[:split_idx]
            class_name = path[split_idx + 1:]
            module = importlib.import_module(import_path)
            class_ = getattr(module, class_name)
            obj = class_(**curr_params)

        return copy.deepcopy(obj)














