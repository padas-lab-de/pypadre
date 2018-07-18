from padre.wrappers.wrapper_pytorch import  WrapperPytorch
from sklearn.pipeline import  Pipeline
from sklearn import datasets
import torch
import numpy as np

def main():
    layers = []
    layers.append(torch.nn.Linear(3, 20))
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
    workflow.fit(np.asarray(x), np.reshape(y, newshape=(150,1)))
    print('Fit completed')
    y_pred = workflow.infer()
    print('Infering completed')




if __name__ == '__main__':
    main()