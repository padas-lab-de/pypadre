from padre.wrappers.wrapper_pytorch import  WrapperPytorch
from padre.ds_import import load_sklearn_toys
from sklearn.pipeline import  Pipeline
from sklearn import datasets
import torch
import numpy as np
from padre.experiment import Experiment, Splitter
from padre.app import pypadre

def main():
    layers = []
    layers.append(torch.nn.Linear(4, 20))
    layers.append(torch.nn.ReLU())
    layers.append(torch.nn.Linear(20, 10))
    layers.append(torch.nn.Linear(10, 1))

    params = dict()
    params['lr'] = 0.01

    obj = WrapperPytorch(layers, params)
    estimators = [('clf', obj)]
    workflow = Pipeline(estimators)
    iris = datasets.load_iris()
    x = iris.data[:, :3]
    y = iris.target
    ds = [i for i in load_sklearn_toys()][4]
    #workflow.fit(np.asarray(x), np.reshape(y, newshape=(150,1)))
    ex = Experiment(name="Torch",
                    description="Testing Torch via SKLearn Pipeline",
                    dataset=ds,
                    workflow=workflow,
                    backend=pypadre.file_repository.experiments)
    ex.run()




if __name__ == '__main__':
    main()