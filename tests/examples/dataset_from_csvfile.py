"""
This file shows an example on how to use the pypadre app.
"""
from padre.ds_import import load_sklearn_toys
from padre.base import PadreLogger
from padre.eventhandler import add_logger
import pprint
from padre.core import Experiment
from padre.ds_import import load_csv


def create_test_pipeline():
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.decomposition import PCA
    estimators = [('PCA', PCA()), ('SVC', SVC())]
    #estimators = [('k-nn classifier', KNeighborsClassifier())]
    return Pipeline(estimators)

def create_test_pipeline_regression():
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVR

    # estimators = [('reduce_dim', PCA()), ('clf', SVC())]
    estimators = [('SVR', SVR())]
    return Pipeline(estimators)


def split(idx):
    # Do a 70:30 split
    limit = int(.7 * len(idx))
    return idx[0:limit], idx[limit:], None


if __name__ == '__main__':
    from padre.app import pypadre
    pypadre.set_printer(print)

    # NOTE: Server MUST BE RUNNING!!! See Padre Server!
    # Start PADRE Server and run
    ds = load_csv('/home/christofer/PycharmProjects/TwitterCrawler/datasets/merged/twitterbot.csv',
                  targets=['bot'],
                  description='Crawled Twitter data for identifying bots')
    #pypadre.datasets.put(ds, upload=True)
    ex = Experiment(name="Test Experiment SVM",
                    description="Testing Support Vector Machines via SKLearn Pipeline",
                    dataset=ds,
                    workflow=create_test_pipeline(), keep_splits=True, strategy="cv")
    conf = ex.configuration()  # configuration, which has been automatically extracted from the pipeline
    pprint.pprint(ex.hyperparameters())  # get and print hyperparameters
    ex.execute()  # run the experiment and report

    pypadre.metrics_evaluator.add_experiments([ex, ex])
    print(pypadre.metrics_evaluator.show_metrics())
    '''
    import numpy as np
    import pandas as pd
    from padre.ds_import import load_pandas_df
    data = np.random.random_sample((5, 11))
    df = pd.DataFrame(data)
    df.columns = list('abcdefghijk')
    ds = load_pandas_df(df)
    ex = Experiment(name="Test Experiment SVM",
                    description="Testing Support Vector Machines via SKLearn Pipeline",
                    dataset=ds,
                    workflow=create_test_pipeline_regression(), keep_splits=True, strategy="random")
    ex.execute()
    '''
    # ex.report_results() # last step, but we can also look that up on the server

