from pypa_pytorch.pytorch import WrapperPytorch, CallBack
from padre.ds_import import load_sklearn_toys
from sklearn.pipeline import Pipeline
from sklearn import datasets
import torch
from padre.core import Experiment
from padre.app import pypadre


class TestCallbacks(CallBack):
    def on_compute_loss(self, loss):
        print('Function on compute loss. Loss = {loss}'.format(loss=loss))

    def on_epoch_end(self, obj):
        print('Epoch ended')

    def on_epoch_start(self, obj):
        print('Epoch started')

    def on_iteration_start(self, obj):
        print('Iteration started')

    def on_iteration_end(self, obj):
        print('Iteration ended')


def main():
    layers = []
    layers.append(torch.nn.Linear(4, 20))
    layers.append(torch.nn.ReLU())
    layers.append(torch.nn.Linear(20, 10))
    layers.append(torch.nn.Linear(10, 1))

    params = dict()
    params['lr'] = 0.01

    import json
    with open('classification.json') as json_data:
        params = json.load(json_data)

    obj = WrapperPytorch(params=params)
    estimators = [('pytorch', obj)]
    workflow = Pipeline(estimators)
    obj.set_callbacks([TestCallbacks()])
    iris = datasets.load_iris()
    x = iris.data[:, :3]
    y = iris.target
    ds = [i for i in load_sklearn_toys()][4]
    #workflow.fit(np.asarray(x), np.reshape(y, newshape=(150,1)))
    ex = Experiment(name="Torch",
                    description="Testing Torch via SKLearn Pipeline",
                    dataset=ds,
                    workflow=workflow, keep_splits=True, strategy='cv')
    ex.execute()
    '''
    Sample network dictionary creation
    
    layer1 = dict()
    layer1['type'] = 'linear'
    param = dict()
    param['in_features'] = 4
    param['out_features'] = 20
    param['bias'] = True
    layer1['params'] = copy.deepcopy(param)

    layer2 = dict()
    layer2['type'] = 'relu'

    layer3 = dict()
    layer3['type'] = 'linear'
    param = dict()
    param['in_features'] = 20
    param['out_features'] = 10
    param['bias'] = True
    layer3['params'] = copy.deepcopy(param)

    layer4 = dict()
    param = dict()
    layer4['type'] = 'linear'
    param['in_features'] = 10
    param['out_features'] = 1
    param['bias'] = True
    layer4['params'] = copy.deepcopy(param)

    layers = dict()
    layers['layer1'] = layer1
    layers['layer2'] = layer2
    layers['layer3'] = layer3
    layers['layer4'] = layer4
    
    layer_order = ['layer1', 'layer2', 'layer3', 'layer4']
    
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
    with open('config.json', 'w') as fp:
        json.dump(network, fp)
    
    '''

if __name__ == '__main__':
    main()
