"""
This file tests the different learning methods and its parameters
"""
import pprint

from sklearn import linear_model, decomposition
from sklearn.svm import SVC

from padre.ds_import import load_sklearn_toys
from padre.experiment import Experiment


def get_local_dataset(name=None):
    local_dataset = ['Boston_House_Prices',
                     'Breast_Cancer',
                     'Diabetes',
                     'Digits',
                     'Iris',
                     'Linnerrud']
    if name is None:
        print('Dataset name is empty')
        return None
    if name in local_dataset:
        return [i for i in load_sklearn_toys()][local_dataset.index(name)]
    else:
        print(name + ' Local Dataset not found')
        return None


# TODO: This should mapped dynamically
# TODO: Parameters should be accepted to the workflow
workflow_dict = {'pca': decomposition.PCA(),
                 'logistic': linear_model.LogisticRegression(),
                 'SVC': SVC(probability=True),
                 'LSA': decomposition.TruncatedSVD(n_components=2)}


def create_test_pipeline(estimator_list):
    from sklearn.pipeline import Pipeline
    estimators = []
    # estimators=[('pca', decomposition.PCA()), ('logistic', linear_model.LogisticRegression())]
    for estimator in estimator_list:
        if workflow_dict.get(estimator) is None:
            print(estimator + ' not present in list')
            return None
        estimators.append((estimator, workflow_dict.get(estimator)))
    return Pipeline(estimators)


experiments_dict = dict()


def create_experiment(name, description,
                      dataset, workflow,
                      backend):
    # Experiment name should be unique
    if experiments_dict.get(name, None) is None:
        data_dict = dict()
        data_dict['description'] = description
        data_dict['dataset'] = dataset
        data_dict['workflow'] = workflow
        data_dict['backend'] = backend
        experiments_dict[name] = data_dict

    else:
        print('Experiment name should be unique')
        print(name + ' already exists in experiments')


def main():
    # Get the dataset from load_sklearn_toys
    # The dataset index is 3
    from padre.app import pypadre
    pypadre.set_printer(print)
    # NOTE: Server MUST BE RUNNING!!! See Padre Server!
    # Start PADRE Server and run
    ds = None
    try:
        pypadre.datasets.list_datasets()
        ds = pypadre.datasets.get_dataset("http://localhost:8080/api/datasets/5")
    except:
        ds = [i for i in load_sklearn_toys()][2]

    # ex.report_results() # last step, but we can also look that up on the server

    # Create a sample experiment
    workflow = create_test_pipeline(['SVC'])
    create_experiment(name='Experiment1', description='This is the first test experiment',
                      dataset=ds, workflow=workflow,
                      backend=pypadre.file_repository.experiments)

    # Create another experiment which has a transformation and logistic regression
    workflow = create_test_pipeline(['pca', 'logistic'])
    create_experiment(name='Experiment2', description='This is the second test experiment',
                      dataset=[i for i in load_sklearn_toys()][3], workflow=workflow,
                      backend=pypadre.file_repository.experiments)
    workflow = create_test_pipeline(['LSA'])
    create_experiment(name='Experiment3', description='This is the third test experiment',
                      dataset=get_local_dataset('Diabetes'), workflow=workflow,
                      backend=pypadre.file_repository.experiments)

    # Run all the experiments in the list
    for experiment in experiments_dict:
        ex = Experiment(name=experiment,
                        description=experiments_dict.get(experiment).get('description'),
                        dataset=experiments_dict.get(experiment).get('dataset', None),
                        workflow=experiments_dict.get(experiment).get('workflow', None),
                        backend=experiments_dict.get(experiment).get('backend', None))
        conf = ex.configuration()  # configuration, which has been automatically extracted from the pipeline
        pprint.pprint(ex.hyperparameters())  # get and print hyperparameters
        ex.run()  # run the experiment and report

    print("========Available experiments=========")
    for idx, ex in enumerate(pypadre.experiments.list_experiments()):
        print("%d: %s" % (idx, str(ex)))
        for idx2, run in enumerate(pypadre.experiments.list_runs(ex)):
            print("\tRun: %s" % str(run))


if __name__ == '__main__':
    main()
