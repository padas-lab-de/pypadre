"""
This file shows an example on how to use the pypadre app.
"""
from padre.experiment import Experiment
import sklearn.model_selection as ms
import pprint

def create_test_pipeline():
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    from sklearn.decomposition import PCA
    estimators = [('reduce_dim', PCA()), ('clf', SVC())]
    return Pipeline(estimators)



if __name__ == '__main__':
    from padre.app import pypadre
    pypadre.set_printer(print)
    pypadre.datasets.list()
    ds = pypadre.datasets.get("http://localhost:8080/api/datasets/3")
    print(ds)
    ms.cross_val_predict()
    ex = Experiment(ds,
                    create_test_pipeline(),
                    splitting_strategy={
                        "name": "random",
                        "n_folds": 10,
                        "test_size": 0.4
                    },
                    random_state=0)
    conf = ex.configuration()  # configuration, which has been automatically extracted from the pipeline
    pprint.pprint(ex.hyperparameters())  # get and print hyperparameters
    ex.run()  # run the experiment and report
    #ex.report_results() # last step, but we can also look that up on the server