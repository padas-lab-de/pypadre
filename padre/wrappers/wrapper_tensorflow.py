import torch
import copy
import numpy as np
import json
import importlib
import os
import random

class WrapperTensorFlow:
    """
    This class wraps the whole Tensorflow network and exposes an easy to use interface to the user
    """

    # The different dictionaries containing information about the objects, its parameters etc
    layers_dict = None
    transforms_dict = None
    optimizers_dict = None
    loss_dict = None
    lr_scheduler_dict = None

    # Model related variables
    model = None
    lr_scheduler = None
    optimizer = None

    transforms = None

    top_shape = 0

    probabilities = None

    flatten = False

    checkpoint = 0

    resume = False

    pre_trained_model_path = None

    learning_rate = None

    classification = False

    def __init__(self, params:dict):
        """
        Function for initializing the wrapper, creating the network and setting all the necessary parameters

        :param params: Parameters of the network, shape of the network
        """

        with open('mappings_tensorflow.json') as f:
            framework_dict = json.load(f)

        self.layers_dict = framework_dict.get('layers', None)

        if self.layers_dict is None:
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

        if shape is None:
            return

        if self.learning_rate is None:
            self.learning_rate = 0.1

        self.model = shape

    def fit(self, x, y):
        """
        This function creates a model from the existing data

        :param x: Training data
        :param y: Traning labels/values

        :return: None
        """

        print('Fit')

        import tensorflow as tf



        step = 0

        x = tf.cast(x, tf.float32)
        X = tf.placeholder("float", [None, x.shape[1]])

        if self.classification is True:
            Y = tf.placeholder("float", [None, self.top_shape])
            y = tf.one_hot(y, 3)

        else:
            y = np.reshape(y, newshape=(y.shape[0], 1))
            Y = tf.placeholder("float", [None, 1])

        # TODO: Implement batch processing
        x_mini_batch = x
        y_mini_batch = y

        results = self.execute_model(x_mini_batch)
        prediction = tf.nn.softmax(results)
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=results, labels=y_mini_batch))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train_op = optimizer.minimize(loss_op)

        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_mini_batch, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            while step < self.steps:
                print(step)
                sess.run([loss_op, accuracy], feed_dict={X:x_mini_batch.eval(), Y:y_mini_batch.eval()})
                step = step + 1

    def predict(self, x):
        """
        This function predicts based on the model created in fit

        :param x: Testing data

        :return: Results
        """

        print('Predict')
        import tensorflow as tf

        step = 0

        x = tf.cast(x, tf.float32)
        # y = tf.one_hot(y, 3)
        X = tf.placeholder("float", [None, x.shape[1]])

        # TODO: Implement batch processing
        x_mini_batch = x

        results = self.execute_model(x_mini_batch)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print(step)
            sess.run([results], feed_dict={X: x_mini_batch.eval()})
            predicted = results.eval()

        if self.classification is True:
            self.probabilities = predicted
            predicted = predicted.argmax(axis=1)

        return predicted

    def predict_proba(self, x):
        """
        Returns the predicted probabilities for a classification model.
        Currently, this returns the previous probabilities stored inorder to avoid recomputation

        :param x: The input vectors

        :return: The probabilities of the predictions
        """

        if self.probabilities is None:
            probabilites = np.zeros(shape=(len(x), self.top_shape))

        else:
            probabilities = self.probabilities

        return probabilities


    def execute_model(self, x):
        """
        This function runs the input through the model

        :param x: Input feature vectors

        :return: results of the pass
        """

        result = x

        for layer in self.model:
            print(layer)
            result = layer(result)

        return result

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
                if str(layer_type).upper() in ['DENSE']:
                    self.top_shape = params.get('units')
                    if self.top_shape > 1:
                        self.classification = True

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
            print(param)
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

    def create_loss_function(self, name, params):
        """
        Function to create the loss function and set all the corresponding params in a dictionary

        :param params: Parameters of the loss function

        :return: The loss function and a dictionary containing all the required parameters
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

                loss_params = loss_function_details.get('params', None)
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
                loss = getattr(module, class_name)

                if curr_params is None:
                    curr_params = dict()

        return copy.deepcopy(loss), copy.deepcopy(curr_params)