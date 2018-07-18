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

        self.lr = params.get('lr', 0.001)
        self.steps = params.get('steps', 1000)
        self.shape = shape
        self.loss_function = params.get('loss_function', 'MSELoss')
        self.batch_size = params.get('batch_size', 1)
        self.model = self.create_model(shape)
        self.optimizer = self.create_optimizer(params, self.lr)
        self.params = params
        self.loss = self.create_loss()

    def fit(self, x, y):

        print('fit')
        x = torch.autograd.Variable(torch.from_numpy(x), requires_grad=False)
        y = torch.autograd.Variable(torch.DoubleTensor(y), requires_grad=False)
        self.model = self.model.double()
        for step in range(self.steps):
            y_pred = self.model(x)
            loss = self.loss(y_pred, y)
            print(step, loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def infer(self, x):

        return self.model(x)

    def create_model(self, shape):

        model = torch.nn.Sequential(*shape)
        return model

    def create_optimizer(self, params, lr):
        """
        Reference: https://pytorch.org/docs/stable/optim.html

        :param params: A dictionary containing the parameters required for the optimizer
        :param lr: The learning rate

        :return: Optimizer object
        """

        optimizer_name = str(params.get('optimizer', 'Adam')).upper()

        optimizer = None

        if optimizer_name == 'ADADELTA':
            rho = self.params.get('rho', 0.9)
            eps = self.params.get('eps', 0.000001)
            weight_decay = self.params.get('weight_decay', 0)

        elif optimizer_name == 'ADAGRAD':
            lr_decay = self.params.get('lr_decay', 0)
            weight_decay = self.params.get('weight_decay', 0)

        elif optimizer_name == 'ADAM':
            #betas = self.params.get('')
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        elif optimizer_name == 'SGD':
            momentum = self.params.get('momentum', 0.9)
            optimizer = torch.optim.SGD(self.model.parameters(), self.lr, momentum)

        else:
            momentum = self.params.get('momentum', 0.9)
            lr = self.params.get('lr', 0.1)
            optimizer = torch.optim.SGD(self.model.parameters(), lr, momentum)

        return optimizer

    def create_loss(self):
        return torch.nn.MSELoss(size_average=False)






