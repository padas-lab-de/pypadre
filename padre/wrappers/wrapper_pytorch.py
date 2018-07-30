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
# TODO: Implement Recurrent Layers
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
            output = self.model(x)

        return output

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

        for layer in layer_order:

            layer = architecture.get(layer, None)
            if layer is None:
                return None

            layer_type = layer.get('type')
            params = layer.get('params')

            layer_obj = self.create_layer_object(layer_type, params)

            if layer_obj is not None:
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

        layer_obj = None

        # Convolutional layers
        if layer_type == 'CONV1D':
            if self.verify_convolutional_params(layer_params):
                layer_obj = torch.nn.Conv1d(**layer_params)

        elif layer_type == 'CONV2D':
            if self.verify_convolutional_params(layer_params):
                layer_obj = torch.nn.Conv2d(**layer_params)

        elif layer_type == 'CONV3D':
            if self.verify_convolutional_params(layer_params):
                layer_obj = torch.nn.Conv3d(**layer_params)

        # Transpose Layers
        elif layer_type == 'CONVTRANSPOSE1D':
            if self.verify_convolutional_params(layer_params):
                layer_obj = torch.nn.ConvTranspose1d(**layer_params)

        elif layer_type == 'CONVTRANSPOSE2D':
            if self.verify_convolutional_params(layer_params):
                layer_obj = torch.nn.ConvTranspose2d(**layer_params)

        elif layer_type == 'CONVTRANSPOSE3D':
            if self.verify_convolutional_params(layer_params):
                layer_obj = torch.nn.ConvTranspose3d(**layer_params)

        elif layer_type == 'UNFOLD':
            layer_obj = torch.nn.Unfold(**layer_params)

        elif layer_type == 'FOLD':
            layer_obj = torch.nn.Fold(**layer_params)

        # Max Pooling Layers
        elif layer_type == 'MAXPOOL1D':
            if self.verify_pooling_params(layer_params):
                layer_obj = torch.nn.MaxPool1d(**layer_params)

        elif layer_type == 'MAXPOOL2D':
            if self.verify_pooling_params(layer_params):
                layer_obj = torch.nn.MaxPool2d(**layer_params)

        elif layer_type == 'MAXPOOL3D':
            if self.verify_pooling_params(layer_params):
                layer_obj = torch.nn.MaxPool3d(**layer_params)

        # Unpooling Layers
        elif layer_type == 'MAXUNPOOL1D':
            if self.verify_pooling_params(layer_params):
                layer_obj = torch.nn.MaxUnpool1d(**layer_params)

        elif layer_type == 'MAXUNPOOL2D':
            if self.verify_pooling_params(layer_params):
                layer_obj = torch.nn.MaxUnpool2d(**layer_params)

        elif layer_type == 'MAXUNPOOL3D':
            if self.verify_pooling_params(layer_params):
                layer_obj = torch.nn.MaxUnpool3d(**layer_params)

        # Average Pooling Layers
        elif layer_type == 'AVGPOOL1D':
            if self.verify_pooling_params(layer_params):
                layer_obj = torch.nn.AvgPool1d(**layer_params)

        elif layer_type == 'AVGPOOL2D':
            if self.verify_pooling_params(layer_params):
                layer_obj = torch.nn.AvgPool2d(**layer_params)

        elif layer_type == 'AVGPOOL3D':
            if self.verify_pooling_params(layer_params):
                layer_obj = torch.nn.AvgPool3d(**layer_params)

        # Fractional Pooling
        elif layer_type == 'FRACTIONALMAXPOOL2D':
            if self.verify_pooling_params(layer_params):
                layer_obj = torch.nn.FractionalMaxPool2d(**layer_params)

        # Power Average Pooling
        elif layer_type == 'LPPOOL1D':
            if self.verify_pooling_params(layer_params) and \
                    layer_params.get('norm_type', None) is not None:
                layer_obj = torch.nn.LPPool1d(**layer_params)

        elif layer_type == 'LPPOOL2D':
            if self.verify_pooling_params(layer_params) and \
                    layer_params.get('norm_type', None) is not None:
                layer_obj = torch.nn.LPPool2d(**layer_params)

        # Adaptive Pooling Layers
        elif layer_type == 'ADAPTIVEMAXPOOL1D':
            if layer_params.get('output_size', None) is not None:
                layer_obj = torch.nn.AdaptiveMaxPool1d(**layer_params)

        elif layer_type == 'ADAPTIVEMAXPOOL2D':
            if layer_params.get('output_size', None) is not None:
                layer_obj = torch.nn.AdaptiveMaxPool2d(**layer_params)

        elif layer_type == 'ADAPTIVEMAXPOOL3D':
            if layer_params.get('output_size', None) is not None:
                layer_obj = torch.nn.AdaptiveMaxPool3d(**layer_params)

        elif layer_type == 'ADAPTIVEAVGPOOL1D':
            if layer_params.get('output_size', None) is not None:
                layer_obj = torch.nn.AdaptiveAvgPool1d(**layer_params)

        elif layer_type == 'ADAPTIVEAVGPOOL2D':
            if layer_params.get('output_size', None) is not None:
                layer_obj = torch.nn.AdaptiveAvgPool2d(**layer_params)

        elif layer_type == 'ADAPTIVEAVGPOOL3D':
            if layer_params.get('output_size', None) is not None:
                layer_obj = torch.nn.AdaptiveAvgPool3d(**layer_params)

        # Reflection Padding Layers
        elif layer_type == 'REFLECTIONPAD1D':
            if self.verify_padding_params(layer_params):
                layer_obj = torch.nn.ReflectionPad1d(**layer_params)

        elif layer_type == 'REFLECTIONPAD2D':
            if self.verify_padding_params(layer_params):
                layer_obj = torch.nn.ReflectionPad2d(**layer_params)

        # Replication Padding Layers
        elif layer_type == 'REPLICATIONPAD1D':
            if self.verify_padding_params(layer_params):
                layer_obj = torch.nn.ReplicationPad1d(**layer_params)

        elif layer_type == 'REPLICATIONPAD2D':
            if self.verify_padding_params(layer_params):
                layer_obj = torch.nn.ReplicationPad2d(**layer_params)

        elif layer_type == 'REPLICATIONPAD3D':
            if self.verify_padding_params(layer_params):
                layer_obj = torch.nn.ReplicationPad3d(**layer_params)

        # Zero Padding Layers
        elif layer_type == 'ZEROPAD2D':
            if self.verify_padding_params(layer_params):
                layer_obj = torch.nn.ZeroPad2d(**layer_params)

        # Constant Padding Layers
        elif layer_type == 'CONSTANTPAD1D':
            if self.verify_padding_params(layer_params) and \
                    layer_params.get('value', None) is not None:
                layer_obj = torch.nn.ConstantPad1d(**layer_params)

        elif layer_type == 'CONSTANTPAD2D':
            if self.verify_padding_params(layer_params) and \
                    layer_params.get('value', None) is not None:
                layer_obj = torch.nn.ConstantPad2d(**layer_params)

        elif layer_type == 'CONSTANTPAD3D':
            if self.verify_padding_params(layer_params) and \
                    layer_params.get('value', None) is not None:
                layer_obj = torch.nn.ConstantPad3d(**layer_params)

        # Non Linear Activation Layers
        elif layer_type == 'ELU':
            layer_obj = torch.nn.ELU(**layer_params)

        elif layer_type == 'HARDSHRINK':
            layer_obj = torch.nn.Hardshrink(**layer_params)

        elif layer_type == 'HARDTANH':
            layer_obj = torch.nn.Hardtanh(**layer_params)

        elif layer_type == 'LEAKYRELU':
            layer_obj = torch.nn.LeakyReLU(**layer_params)

        elif layer_type == 'LOGSIGMOID':
            layer_obj = torch.nn.LogSigmoid()

        elif layer_type == 'PRELU':
            if layer_params.get('num_parameters', None) is not None and \
                    layer_params.get('init', None) is not None:
                layer_obj = torch.nn.PReLU(**layer_params)

        elif layer_type == 'RELU':
            layer_obj = torch.nn.ReLU(**layer_params)

        elif layer_type == 'RELU6':
            layer_obj = torch.nn.ReLU6(**layer_params)

        elif layer_type == 'RRELU':
            layer_obj = torch.nn.RReLU(**layer_params)

        elif layer_type == 'SELU':
            layer_obj = torch.nn.SELU(**layer_params)

        elif layer_type == 'SIGMOID':
            layer_obj = torch.nn.SELU()

        elif layer_type == 'SOFTPLUS':
            layer_obj = torch.nn.Softplus(**layer_params)

        elif layer_type == 'SOFTSHRINK':
            layer_obj = torch.nn.Softshrink(**layer_params)

        elif layer_type == 'SOFTSIGN':
            layer_obj = torch.nn.Softsign()

        elif layer_type == 'TANH':
            layer_obj = torch.nn.Tanh()

        elif layer_type == 'TANHSHRINK':
            layer_obj = torch.nn.Tanhshrink()

        elif layer_type == 'THRESHOLD':
            if layer_params.get('threshold', None) is not None and \
                    layer_params.get('value', None) is not None:
                layer_obj = torch.nn.Threshold(**layer_params)

        elif layer_type == 'SOFTMIN':
            layer_obj = torch.nn.Softmin(**layer_params)

        elif layer_type == 'SOFTMAX':
            layer_obj = torch.nn.Softmax(**layer_params)

        elif layer_type == 'SOFTMAX2D':
            layer_obj = torch.nn.Softmax2d()

        elif layer_type == 'LOGSOFTMAX':
            layer_obj = torch.nn.LogSoftmax(**layer_params)

        # Batch Normalization Layers
        elif layer_type == 'BATCHNORM1D':
            if self.verify_batch_norm_params(layer_params):
                layer_obj = torch.nn.BatchNorm1d(**layer_params)

        elif layer_type == 'BATCHNORM2D':
            if self.verify_batch_norm_params(layer_params):
                layer_obj = torch.nn.BatchNorm2d(**layer_params)

        elif layer_type == 'BATCHNORM3D':
            if self.verify_batch_norm_params(layer_params):
                layer_obj = torch.nn.BatchNorm3d(**layer_params)

        elif layer_type == 'GROUPNORM':
            if layer_params.get('num_groups', None) is not None and \
                    layer_params.get('num_channels', None) is not None:
                layer_obj = torch.nn.GroupNorm(**layer_params)

        elif layer_type == 'INSTANCENORM1D':
            if self.verify_instance_norm_params(layer_params):
                layer_obj = torch.nn.InstanceNorm1d(**layer_params)

        elif layer_type == 'INSTANCENORM2D':
            if self.verify_instance_norm_params(layer_params):
                layer_obj = torch.nn.InstanceNorm2d(**layer_params)

        elif layer_type == 'INSTANCENORM3D':
            if self.verify_instance_norm_params(layer_params):
                layer_obj = torch.nn.InstanceNorm3d(**layer_params)

        elif layer_type == 'LOCALRESPONSENORM':
            if layer_params.get('size', None) is not None:
                layer_obj = torch.nn.LocalResponseNorm(**layer_params)

        # Linear Layers
        elif layer_type == 'LINEAR':
            if self.verify_linear_params(layer_params):
                layer_obj = torch.nn.Linear(**layer_params)

        elif layer_type == 'BILINEAR':
            if self.verify_bilinear_params(layer_params):
                layer_obj = torch.nn.Bilinear(**layer_params)

        # Dropout Layers
        elif layer_type == 'DROPOUT':
            layer_obj = torch.nn.Dropout(**layer_params)

        elif layer_type == 'DROPOUT2D':
            layer_obj = torch.nn.Dropout2d(**layer_params)

        elif layer_type == 'DROPOUT3D':
            layer_obj = torch.nn.Dropout3d(**layer_params)

        elif layer_type == 'ALPHADROPOUT':
            layer_obj = torch.nn.AlphaDropout(**layer_params)

        # Sparse Layers
        elif layer_type == 'EMBEDDING':
            if self.verify_embedding_params(layer_params):
                layer_obj = torch.nn.Embedding(**layer_params)

        elif layer_type == 'EMBEDDINGBAG':
            if self.verify_embedding_params(layer_params):
                layer_obj = torch.nn.EmbeddingBag(**layer_params)

        else:
            layer_obj = None

        '''
        The following function isn't present in the library but is present in the documentation of Torch
        elif layer_type == 'ADAPTIVELOGSOFTMAXWITHLOSS':
            if self.verify_adaptivesoftmaxwithloss_params(layer_params):
                layer_obj = torch.nn.AdaptiveLogSoftmaxwithLoss(**layer_params)
        '''

        if layer_obj is not None:
            return copy.deepcopy(layer_obj)

        else:
            return None

    def verify_convolutional_params(self, params=None):
        """
        The function verifies the paramaters to be passed for creating convolutional layers

        :param params: The parameters for creating the convolutional layer

        :return: True if successful, False otherwise

        TODO: Add datatype validation for arguments and also verify that no extra arguments are present in params
        """
        flag = True

        if params is None:
            params = dict()

        in_channels = params.get('in_channels', None)
        out_channels = params.get('out_channels', None)
        kernel_size = params.get('kernel_size', None)

        if in_channels is None or out_channels is None or kernel_size is None:
            flag = False

        return flag

    def verify_pooling_params(self, params=None):
        """
        The function verifies the pooling parameters to be passed to the layer constructor.

        :param params: params to be input to the torch pooling functions.

        :return: True if successful, False otherwise.
        """

        flag = True
        if params is None:
            params = dict()

        kernel_size = params.get('kernel_size', None)

        if kernel_size is None:
            flag = False

        return flag

    def verify_padding_params(self, params=None):
        """
        The function verifies the padding parameters to be passed to the layer constructor.

        :param params: Params to be input to the torch padding functions.

        :return: True if successful, False otherwise
        """

        flag = True
        if params is None:
            params = dict()

        padding = params.get('padding', None)

        if padding is None:
            flag = False

        return flag

    def verify_adaptivesoftmaxwithloss_params(self, params=None):
        """
        The function verifies the parameters to be passed to the Adaptive Softmax with loss Layer

        :param params: Parameters to be input to the Adaptive Softmax with Loss function

        :return: True if successful, False otherwise
        """

        if params is None:
            params = dict()

        flag = True

        in_features = params.get('in_features', None)
        n_classes = params.get('n_classes', None)
        cutoffs = params.get('cutoffs')

        if in_features is None or n_classes is None or cutoffs is None:
            flag = False

        return flag

    def verify_batch_norm_params(self, params=None):
        """
        The function verifies the parameters to be passed to the Batch Normalization layers

        :param params: Parameters to be input to the Batch Normalization Layer

        :return: True if successful, False otherwise
        """

        if params is None:
            params = dict()

        flag = True

        num_features = params.get('num_features', None)

        if num_features is None:
            flag = False

        return flag

    def verify_instance_norm_params(self, params=None):


        if params is None:
            params = dict()

        flag = True

        num_features = params.get('num_features', None)

        if num_features is None:
            flag = False

        return flag

    def verify_linear_params(self, params=None):
        """
        The functions verifies the input arguments for the Linear Layer

        :param params: Params to be used as input for the Linear Layer

        :return: True if successful, False otherwise
        """

        if params is None:
            params = dict()

        flag = True

        in_features = params.get('in_features', None)
        out_features = params.get('out_features')

        if in_features is None or out_features is None:
            flag = False

        return flag

    def verify_bilinear_params(self, params=None):
        """
        The function verifies the bilinear layer parameters

        :param params: Parameters to be used to create a bilinear layers

        :return: True if successful, False otherwise
        """

        if params is None:
            params = dict()

        flag = True

        in1_features = params.get('in1_features', None)
        in2_features = params.get('in2_features', None)
        out_features = params.get('out_features', None)

        if in1_features is None or in2_features is None or out_features is None:
            flag = False

        return flag

    def verify_embedding_params(self, params=None):
        """
        This function verifies the parameters for the Embedding Layer in torch

        :param params: Parameters to be passed to the constructor of the embedding layer

        :return: True if successful, False otherwise
        """

        if params is None:
            params = dict()

        flag = True

        num_embeddings = params.get('num_embeddings', None)
        embedding_dim = params.get('embedding_dim', None)

        if num_embeddings is None or embedding_dim is None:
            flag = False

        return flag













