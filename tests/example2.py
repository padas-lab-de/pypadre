"""
This file shows an example on how to use the pypadre app.
"""
from padre.experiment import Experiment
import pprint

from sklearn import svm, datasets
from sklearn.model_selection import cross_val_score
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf = svm.SVC(probability=True, random_state=0)
cross_val_score(clf, X, y, scoring='neg_log_loss')

model = svm.SVC()
cross_val_score(model, X, y, scoring='wrong_choice')

ex = Experiment()

# todo : the exmaple does not work yet, needs more on decarator and syntax conventions.

@ex.workflow
def create_test_pipeline():
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    #from sklearn.decomposition import PCA
    # estimators = [('reduce_dim', PCA()), ('clf', SVC())]
    estimators = [('clf', SVC())]
    return Pipeline(estimators)

@ex.dataset
def get_dataset():
    from padre.app import pypadre
    pypadre.set_printer(print)
    pypadre.datasets.list_datasets()
    return pypadre.datasets.get_dataset("http://localhost:8080/api/datasets/5")



if __name__ == '__main__':

    pprint.pprint(ex.hyperparameters())  # get and print hyperparameters
    ex.run()  # run the experiment and report
