import torch
import copy
import numpy as np
import json
import importlib
import os
import torchvision
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
    transforms_dict = None
    optimizers_dict = None
    loss_dict = None

    transforms = None

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
            framework_dict = json.load(f)
        self.layers_dict = framework_dict.get('layers', None)
        self.transforms_dict = framework_dict.get('transforms', None)
        self.optimizers_dict = framework_dict.get('optimizers', None)
        self.loss_dict = framework_dict.get('loss_functions', None)

        if self.layers_dict is None or self.transforms_dict is None or \
                self.optimizers_dict is None or self.loss_dict is None:
            return

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


        transformers_ = self.params.get('transforms', None)
        transform_order = self.params.get('transform_order', None)

        if transformers_ is not None and transform_order is not None:
            self.create_transforms(transformers=transformers_, transform_order=transform_order)

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
                # Do one hot encoding if it is a classification problem
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

        if self.transforms is not None:
            x = self.transforms(x)

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

        # Load the model if resume is true and the file exists
        if self.resume is True and os.path.isfile(self.pre_trained_model_path):
            state = torch.load(self.pre_trained_model_path)
            self.optimizer.load_state_dict(state['optimizer'])
            self.model.load_state_dict(state['model'])
            step = state['step']

        while step < self.steps:

            if randomize is True:
                permutation = torch.randperm(x.size()[0])

            indices = permutation[start_idx: start_idx + batch_size]

            # Randomize the order after every epoch
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
        # Default optimizer is SGD if no match is found.

        optimizer_dict = self.optimizers_dict.get(optimizer_type, None)
        if optimizer_dict is None:
            optimizer_type = 'SGD'
            optimizer_dict = self.optimizers_dict.get(optimizer_type, None)

        optimizer_params = optimizer_dict.get('params', None)
        if optimizer_params is None:
            optimizer_params = dict()

        curr_params = dict()
        # Iterate through all the parameters possible for the optimizer and select those parameters that are valid
        for param in optimizer_params:
            # Get the corresponding parameter value from the input param list
            param_value = params.get(param, None)
            if param == 'params':
                curr_params[param] = self.model.parameters()

            elif param_value is None and optimizer_params.get(param).get('optional') is False:
                curr_params = None
                break

            else:
                if param_value is not None:
                    curr_params[param] = param_value

        path = optimizer_dict.get('path', None)
        # Dynamically load the module from the path
        if path is not None:
            split_idx = path.rfind('.')
            import_path = path[:split_idx]
            class_name = path[split_idx + 1:]
            module = importlib.import_module(import_path)
            class_ = getattr(module, class_name)
            optimizer = class_(**curr_params)

        return copy.deepcopy(optimizer)

    def create_loss(self, name='MSELOSS', params=None):
        """
        The function returns an object of the required loss function

        :param name: Name of the loss function
        :param params: The parameters of the loss function

        :return: Object of the loss function
        """

        if params is None:
            params = dict()

        loss = None

        name = str(name).upper()

        loss_function_details = self.loss_dict.get(name, None)

        if loss_function_details is not None:
            path = loss_function_details.get('path', None)

            if path is not None:

                curr_params = dict()

                loss_params= loss_function_details.get('params', None)
                if loss_params is None:
                    loss_params = dict()

                for param in loss_params:
                    # Get the corresponding parameter value from the input param list

                    param_value = params.get(param, None)

                    if param_value is None and loss_params.get(param).get('optional') is False:
                        curr_params = None
                        break

                    else:
                        if param_value is not None:
                            curr_params[param] = param_value

                split_idx = path.rfind('.')
                import_path = path[:split_idx]
                class_name = path[split_idx + 1:]
                module = importlib.import_module(import_path)
                class_ = getattr(module, class_name)
                loss = class_(**curr_params)

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


    def create_transforms(self, transformers, transform_order):
        """
        This function creates the necessary transforms to be applied on the data

        :param transforms: The transforms and the corresponding parameters
        :param transform_order: The order in which data transforms should be done

        :return: A transform object if successful, else None
        """

        if transformers is None:
            return None

        transformer_list = []

        for transform in transform_order:
            # Get the transformer object defined in the JSON file
            transformer = self.transforms_dict.get(transform.upper(), None)

            # Get all the possible params from the dictionary
            transform_params = transformer.get('params', None)

            if transform_params is None:
                transform_params = dict()

            # Get all the parameters entered for the transformer
            curr_transformer_params = transformers.get(transform)

            curr_params = dict()
            # Iterate through all the possible params for the transformer
            # This is done so that only the possible parameters are selected to create the object
            for param in transform_params:
                param_value = curr_transformer_params.get(param, None)
                if transform_params is None:
                    continue

                elif param_value is None and transform_params.get(param).get('optional') is False:
                    curr_params = None
                    break

                else:
                    curr_params[param] = param_value

            path = transformer.get('path', None)
            if curr_params is not None and path is not None:
                split_idx = path.rfind('.')
                import_path = path[:split_idx]
                class_name = path[split_idx + 1:]
                module = importlib.import_module(import_path)
                class_ = getattr(module, class_name)
                obj = class_(**curr_params)
                transformer_list.append(copy.deepcopy(obj))

        self.transforms = torchvision.transforms.Compose(transformer_list)