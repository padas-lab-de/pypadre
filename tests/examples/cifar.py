from pypadre.ds_import import load_numpy_array_multidimensional
import numpy as np
import pandas as pd


def main():

    features = np.load('../../features.npy')
    labels = np.load('../../labels.npy')
    ds = load_numpy_array_multidimensional(features=features, targets=labels, columns=['images', 'labels'],
                                           target_features=['labels'])

    print(ds)


if __name__ == '__main__':
    main()
