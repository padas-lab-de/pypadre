import sys
from padre.app import pypadre
pypadre.config.authenticate("cfellicious", "test")

import uuid
import pprint
from padre.core import Experiment

def create_test_pipeline():
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    from sklearn.decomposition import PCA
    # estimators = [('reduce_dim', PCA()), ('clf', SVC())]
    estimators = [('SVC', SVC(probability=True))]
    return Pipeline(estimators)


did = "1"
print("Datasets %s on server" % did)
ds = pypadre.remote_backend.datasets.get(did)
print(ds)
test_experiment_name = "Test Experiment SVM " + str(uuid.uuid4())[0:9] # Unique name for experiment
ex = Experiment(name=test_experiment_name,
                description="Testing Support Vector Machines via SKLearn Pipeline",
                dataset=ds,
                workflow=create_test_pipeline(),
                keep_splits=True, strategy="random")

conf = ex.configuration()  # configuration, which has been automatically extracted from the pipeline
pprint.pprint(ex.hyperparameters())  # get and print hyperparameters
ex.execute()  # run the experiment and report

pypadre.metrics_evaluator.add_experiments([ex])

experiments = pypadre.remote_backend.experiments.list_experiments(test_experiment_name)
experiment_instance = pypadre.remote_backend.experiments.get_experiment("1")
print(experiment_instance)