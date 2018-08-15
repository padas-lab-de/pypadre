from padre.wrappers.wrapper_tensorflow import WrapperTensorFlow
from padre.ds_import import load_sklearn_toys
from sklearn.pipeline import Pipeline
from sklearn import datasets
import torch
import numpy as np
from padre.experiment import Experiment, Splitter
from padre.app import pypadre


def main():

    import json
    with open('config.json') as json_data:
        params = json.load(json_data)

    obj = WrapperTensorFlow(params=params)
    estimators = [('clf', obj)]
    workflow = Pipeline(estimators)
    iris = datasets.load_iris()
    x = iris.data[:, :3]
    y = iris.target
    ds = [i for i in load_sklearn_toys()][4]
    # workflow.fit(np.asarray(x), np.reshape(y, newshape=(150,1)))
    ex = Experiment(name="Torch",
                    description="Testing Torch via SKLearn Pipeline",
                    dataset=ds,
                    workflow=workflow,
                    backend=pypadre.file_repository.experiments)
    ex.run()
    '''
    Sample network dictionary creation

    layer1 = dict()
    layer1['type'] = 'dense'
    param = dict()
    param['units'] = 10
    layer1['params'] = copy.deepcopy(param)

    layer2 = dict()
    layer2['type'] = 'relu'

    layer3 = dict()
    layer3['type'] = 'dense'
    param = dict()
    param['units'] = 20
    param['bias'] = True
    layer3['params'] = copy.deepcopy(param)

    layer4 = dict()
    param = dict()
    layer4['type'] = 'dense'
    param['units'] = 3
    layer4['params'] = copy.deepcopy(param)

    layers = dict()
    layers['layer1'] = layer1
    layers['layer2'] = layer2
    layers['layer3'] = layer3
    layers['layer4'] = layer4

    layer_order = ['layer1', 'layer3', 'layer4']

    optimizer = dict()
    optimizer['type'] = 'SGD'
    params = dict()
    params['momentum'] = 0.9
    params['dampening'] = 0
    params['weight_decay'] = 0
    params['Nesterov'] = False
    params['lr'] = 0.01
    optimizer['params'] = copy.deepcopy(params)

    loss = dict()
    loss['type'] = 'MSELOSS'
    params = dict()
    params['size_average'] = True
    params['reduce'] = True
    loss['params'] = copy.deepcopy(params)


    network = dict()
    network['layer_order'] = layer_order
    network['architecture'] = layers
    network['loss'] = loss
    network['optimizer'] = optimizer
    network['steps'] = 1000
    network['batch_size'] = 1

    import json
    with open('/home/chris/PycharmProjects/PyPaDRe/tests/proof_of_concept/Tensorflow/config.json', 'w') as fp:
        json.dump(network, fp)

    '''


if __name__ == '__main__':
    main()