from pypadre.ds_import import load_numpy_array_multidimensional
import numpy as np
from sklearn.pipeline import Pipeline
from pypa_pytorch import Wrapper
from pypadre.core.experiment import Experiment
from pypadre.app import p_app


def main():

    features = np.load('../../features.npy')
    labels = np.load('../../labels.npy')
    mean_img = np.load('../../mean_file.npy')
    features = np.subtract(features, mean_img)
    ds = load_numpy_array_multidimensional(features=features, targets=labels, columns=['images', 'labels'],
                                           target_features=['labels'])

    print(ds)

    import json
    with open('cifar.json') as json_data:
        params = json.load(json_data)

    obj = Wrapper(params=params)
    estimators = [('pytorch', obj)]
    workflow = Pipeline(estimators)
    # workflow.fit(np.asarray(x), np.reshape(y, newshape=(150,1)))
    ex = Experiment(name="Torch",
                    description="Testing Torch via SKLearn Pipeline",
                    dataset=ds,
                    workflow=workflow, keep_splits=True, strategy='random')
    ex.execute()




if __name__ == '__main__':
    main()
