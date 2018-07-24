from collections import OrderedDict
import torch


class WrapperPytorch:

    model = None

    def __init__(self, shape=None, params=None):
        """
        The initialization function for the pyTorch wrapper
        :param lr: The learning rate for the model
        :param steps: The number of iterations that the training should continue
        :param shape: An array containing the shape of the hidden layers
        :param loss_function: The name of the loss function to be used
        :param batch_size: The batch size used for training
        :param optimizer: The optimizer to be used for training
        """
        print('Initialize')
        if params is None or shape is None:
            return

        self.params = params
        self.lr = params.get('lr', 0.001)
        self.steps = params.get('steps', 1000)
        self.shape = shape
        self.loss_function = params.get('loss_function', 'MSELoss')
        self.batch_size = params.get('batch_size', 1)
        self.model = self.create_model(shape)
        self.optimizer = self.create_optimizer()
        self.params = params
        self.loss = self.create_loss()

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

    def create_optimizer(self):
        """
        Reference: https://pytorch.org/docs/stable/optim.html

        :return: Optimizer object
        """

        lr = self.params.get('lr', 0.001)
        optimizer_name = str(self.params.get('optimizer', 'Adam')).upper()

        optimizer = None

        # Create an object of the optimizer specified by the user.
        # Required parameters are given by the user within the params dictionary.
        # Missing parameters are substituted with default values obtained from the pytorch documentation.
        # Default optimizer is the Adam optimizer and SGD is selected if no match is found.

        if optimizer_name == 'ADADELTA':
            rho = self.params.get('rho', 0.9)
            eps = self.params.get('eps', 0.000001)
            weight_decay = self.params.get('weight_decay', 0)
            optimizer = torch.optim.Adadelta(self.model.parameters(), lr=lr, rho=rho, eps=eps, weight_decay=weight_decay)

        elif optimizer_name == 'ADAGRAD':
            lr_decay = self.params.get('lr_decay', 0)
            weight_decay = self.params.get('weight_decay', 0)
            optimizer = torch.optim.Adagrad(self.model.parameters(), lr=lr, lr_decay=lr_decay, weight_decay=weight_decay)

        elif optimizer_name == 'ADAM':
            betas = self.params.get('betas', (0.9, 0.999))
            eps = self.params.get('eps', 0.00000001)
            weight_decay = self.params.get('weight_decay', 0)
            amsgrad = self.params.get('amsgrad', False)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)

        elif optimizer_name == 'SPARSEADAM':
            betas = self.params.get('betas', (0.9, 0.999))
            eps = self.params.get('eps', 0.00000001)
            optimizer = torch.optim.SparseAdam(self.model.parameters(), lr=lr, betas=betas, eps=eps)

        elif optimizer_name == 'ADAMAX':
            betas = self.params.get('betas', (0.9, 0.999))
            eps = self.params.get('eps', 0.00000001)
            weight_decay = self.params.get('weight_decay', 0)
            optimizer = torch.optim.SparseAdam(self.model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        elif optimizer_name == 'ASGD':
            lambd = self.params.get('lambd', 0.0001)
            alpha = self.params.get('alpha', 0.75)
            t0 = self.params.get('t0', 1000000.0)
            weight_decay = self.params.get('weight_decay', 0)
            optimizer = torch.optim.ASGD(self.model.parameters(), lr=lr, lambd=lambd, alpha=alpha, t0=t0, weight_decay=weight_decay)

        elif optimizer_name == 'LBFGS':
            max_iter = self.params.get('max_iter', 20)
            max_eval = self.params.get('max_eval', None)
            tolerance_grad = self.params.get('tolerance_grad', 0.00001)
            tolerance_change = self.params.get('tolerance_change', 0.000000001)
            history_size = self.params.get('history_size', 100)
            line_search_fn = self.params.get('line_search_fn', None)
            optimizer = torch.optim.LBFGS(self.model.parameters(), lr=lr, max_iter=max_iter,
                                          max_eval=max_eval, tolerance_grad=tolerance_grad,
                                          tolerance_change=tolerance_change, history_size=history_size,
                                          line_search_fn=line_search_fn)

        elif optimizer_name == 'RMSPROP':
            alpha = self.params.get('alpha', 0.75)
            eps = self.params.get('eps', 0.00000001)
            weight_decay = self.params.get('weight_decay', 0)
            momentum = self.params.get('momentum', 0)
            centered = self.params.get('centered', False)
            optimizer = torch.optim.RMSprop(self.model.parameters(), lr=lr, alpha=alpha,
                                            eps=eps, weight_decay=weight_decay,
                                            momentum=momentum, centered=centered)

        elif optimizer_name == 'RPROP':
            etas = tuple(self.params.get('etas', (0.5, 1.2)))
            step_sizes = tuple(self.params.get('step_sizes', (0.000006, 50)))
            optimizer = torch.optim.Rprop(self.model.parameters(), lr=lr, etas=etas, step_sizes=step_sizes)

        elif optimizer_name == 'SGD':
            momentum = self.params.get('momentum', 0.9)
            dampening = self.params.get('dampening', 0)
            weight_decay = self.params.get('weight_decay', 0)
            nesterov = self.params.get('Nesterov', False)
            optimizer = torch.optim.SGD(self.model.parameters(), self.lr, momentum=momentum, dampening=dampening,
                                        weight_decay=weight_decay, nesterov=nesterov)

        else:
            momentum = self.params.get('momentum', 0.9)
            lr = self.params.get('lr', 0.1)
            optimizer = torch.optim.SGD(self.model.parameters(), lr)

        return optimizer

    def create_loss(self, name):
        """
        The function returns an object of the required loss function

        :param name: Name of the loss function

        :return: Object of the loss function
        """

        return torch.nn.MSELoss(size_average=False)






