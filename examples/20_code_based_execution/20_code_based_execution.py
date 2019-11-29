import inspect
from pathlib import Path

from pypadre.binding.model.sklearn_binding import SKLearnPipeline
from pypadre.core.model.code.code_mixin import Function, PythonFile, GitIdentifier
from pypadre.core.model.project import Project
from pypadre.core.model.experiment import Experiment
from pypadre.binding.metrics import sklearn_metrics
from sklearn.datasets import load_iris
from pypadre.examples.base_example import example_app
import numpy as np

print(sklearn_metrics)


# TODO plugin system



def create_test_pipeline_multiple_estimators():
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    from sklearn.decomposition.pca import PCA
    # estimators = [('reduce_dim', PCA()), ('clf', SVC())]
    estimators = [('PCA', PCA()), ('SVC', SVC(probability=True))]
    return Pipeline(estimators)


def main():
    app = example_app()

    # Get the iris dataset and extract the data from it
    data = load_iris().data
    target = load_iris().target.reshape(-1, 1)
    dataset = np.append(data, target, axis=1)

    columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
               'petal width (cm)', 'class']
    target_features = 'class'

    # Use the dataset app to create a PaDRe dataset
    ds = app.datasets.load(source=dataset, name='iris', columns=columns, target_features=target_features)

    # Create a project
    project = Project(name='Example Project',
                      description='Testing the functionalities of project backend',
                      )

    # Create an experiment
    experiment = Experiment(name='Experiment_Code', description='Sample experiment execution using code',
                            dataset=ds, project=project,
                            pipeline=SKLearnPipeline(pipeline_fn=create_test_pipeline_multiple_estimators,
                                                     reference=reference_pipeline),
                            reference=reference_experiment)

    # Add the necessary hyperparameters for grid search
    parameter_dict = {'SVC': {'C': [0.1, 0.2]}, 'PCA': {'n_components': [1, 2, 3]}}
    experiment.execute(parameters={'SKLearnEvaluator': {'write_results': True},
                                   'SKLearnEstimator': {'parameters': parameter_dict}
                                   })

git_repo = str(Path(__file__).parent)
reference_experiment = PythonFile(path=str(Path(__file__).parent), package=__file__[len(git_repo) + 1:],
                       variable=main,
                       repository_identifier=GitIdentifier(path=git_repo))
reference_pipeline = PythonFile(path=str(Path(__file__).parent), package=__file__[len(git_repo) + 1:],
                           variable=create_test_pipeline_multiple_estimators,
                           repository_identifier=GitIdentifier(path=git_repo))
if __name__ == '__main__':
    main()
