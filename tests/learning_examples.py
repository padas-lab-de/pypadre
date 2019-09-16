"""
This file tests the different learning methods and its parameters.
This file also introduces a experimentHelper class that contains
all the parameters, classifiers and estimators for use in the experiment.
"""
import copy
import pprint

import numpy as np
from sklearn import linear_model, decomposition, manifold
from sklearn.svm import SVC, SVR

from pypadre.ds_import import load_sklearn_toys
from pypadre.core import Experiment
from pypadre.logger import PadreLogger
from pypadre.eventhandler import add_logger


# TODO: Method to log errors from outside the experiment class too
class experimentHelper:

    # Initialization of the experiments dict.
    # This dictionary contains all the experiments to be executed.
    # All components of the experiment are contained in this dictionary.
    _experiments = dict()

    # The workflow components contain all the possible components of the workflow.
    # The key is an easy to remember string, and the value is the object of the component.
    # The object is deep copied to form the new component
    _workflow_components = dict()

    # This contains the all the classifiers for checking whether
    # classifiers are given continous data
    _classifier_list = []

    # All parameters of estimators are to be listed here
    # as a list under the key of the estimator name
    # The parameters will be entered as a list to the key
    # based on the estimator name
    # TODO: This should mapped dynamically
    _parameters = dict()

    # All the locally available datasets are mapped to this list.
    _local_dataset = []


    def __init__(self):
        """
        Initialization function of the helper class.
        This function currently manually initializes all values but it could be changed to
        reading the data from files at run time.
        TODO: Make this function read all values dynamically from a file
        """
        
        self._workflow_components = {'pca': decomposition.PCA(),
                                     'logistic': linear_model.LogisticRegression(),
                                     'SVC': SVC(probability=True),
                                     'LSA': decomposition.TruncatedSVD(n_components=2),
                                     'isomap': manifold.Isomap(n_neighbors=10, n_components=5),
                                     'lle': manifold.LocallyLinearEmbedding(n_neighbors=10, n_components=5,
                                                                            method='standard'),
                                     'SVR': SVR(kernel='rbf', degree=4,
                                                gamma='auto', C=1.0)}

        self._classifiers = (['SVC'])

        self._parameters = {'SVC': ['probability'],
                            'LSA': ['n_components'],
                            'isomap': ['n_neighbors', 'n_components'],
                            'lle': ['n_neighbors', 'n_components', 'method'],
                            'SVR': ['kernel', 'degree', 'gamma', 'C']}

        self._local_dataset = ['Boston_House_Prices',
                               'Breast_Cancer',
                               'Diabetes',
                               'Digits',
                               'Iris',
                               'Linnerrud']


    def set_parameters(self, estimator, estimator_name, param_val_dict):
        """
        This function sets the parameters of the estimators and classifiers created
        in the pipeline. The param_val_dict contains the name of the parameter and
        the value to be set for that parameter.
        The parameters are looked up from the parameters dictionary.
        :parameter estimator: The estimator whose parameters have to be modified.
        :parameter estimator_name: The name of the estimator to look up whether,
                                   the parameters are present in that estimator.
        :parameter param_val_dict: The dictionary containing the parameter names
                                   and their values.
        :return If successful, the modified estimator
                Else, None.
        """
        if estimator is None:
            print(estimator_name + ' does not exist in the workflow')
            return None
        available_params = self._parameters.get(estimator_name)
        for param in param_val_dict:
            if param not in available_params:
                print(param + ' is not present for estimator ' + estimator_name)
            else:
                estimator.set_params(**{param: param_val_dict.get(param)})

        return estimator

    def validate_pipeline(self,pipeline):
        """
        This function checks whether each component in the pipeline has a
        fit and fit_transform attribute or transform attribute. The last estimator
        can be none and if it is not none, has to implement a fit attribute.
        :param pipeline: The pipeline of estimators to be created.
        :return: If successful True
                 Else False
        """
        transformers = pipeline[:-1]
        estimator = pipeline[-1]

        for transformer in transformers:
            if (not (hasattr(transformer[1], "fit") or hasattr(transformer[1], "fit_transform")) or not
                     hasattr(transformer[1], "transform")):
                print("All intermediate steps should implement fit and fit_transform or the transform function")
                return False

        if estimator is not None and not(hasattr(estimator[1],"fit")):
            print("Estimator:" + estimator[0]+" does not have attribute fit")
            return  False
        return True


    def create_test_pipeline(self, estimator_list, param_value_dict=None):
        """
        This function creates the pipeline for the experiment.
        The function looks up the name of the estimator passed and then
        deep copies the estimator to the pipeline.
        :param estimator_list: A list of strings containing all estimators to be used
                               in the pipeline in exactly the order to be used
        :param param_value_dict: This contains the parameters for each of the estimators used
                                 The dict is a nested dictionary with the
                                 outer-most key being the estimator name
                                 The inner key is the parameter name
        :return: If successful, the Pipeline containing the estimators
                 Else, None
        """
        from sklearn.pipeline import Pipeline
        estimators = []
        for estimator_name in estimator_list:
            if self._workflow_components.get(estimator_name) is None:
                print(estimator_name + ' not present in list')
                return None

            # Deep copy of the estimator because the estimator object is mutable
            estimator = copy.deepcopy(self._workflow_components.get(estimator_name))
            estimators.append((estimator_name, estimator))
            if param_value_dict is not None and \
                    param_value_dict.get(estimator_name) is not None:
                self.set_parameters(estimator, estimator_name, param_value_dict.get(estimator_name))

        # Check if the created estimators are valid
        if not self.validate_pipeline(estimators):
            return False

        return Pipeline(estimators)

    def create_experiment(self, name, description,
                          dataset, workflow):
        """
        This function adds an experiment to the dictionary.
        :param name: Name of the experiment. It should be unique for this set of experiments
        :param description: The description of the experiment
        :param dataset: The dataset to be used for the experiment
        :param workflow: The scikit pipeline to be used for the experiment.
        :return: None
        """
        if name is None:
            print('Experiment name is missing, a name will be generated by the system')

        if description is None or \
           dataset is None or workflow is None or workflow is False:
            if description is None:
                print('Description is missing for experiment:' + name)
            if dataset is None:
                print('Dataset is missing for experiment:' + name)
            return None

        # Classifiers cannot work on continous data and rejected as experiments.
        if not np.all(np.mod(dataset.targets(), 1) == 0):
            for estimator in workflow.named_steps:
                if estimator in self._classifiers:
                    print('Estimator ' + estimator + ' cannot work on continous data')
                    return None

        # Experiment name should be unique
        if self._experiments.get(name, None) is None:
            data_dict = dict()
            data_dict['description'] = description
            data_dict['dataset'] = dataset
            data_dict['workflow'] = workflow
            self._experiments[name] = data_dict

        else:
            print('Error creating experiment')
            if self._experiments.get(name, None) is not None:
                print('Experiment name: ', name, ' already present. Experiment name should be unique')



    def get_local_dataset(self, name=None):
        """
        This function returns the dataset from pypadre.
        This done by using the pre-defined names of the datasets defined in _local_dataset
        :param name: The name of the dataset
        :return: If successful, the dataset
                 Else, None
        """
        if name is None:
            print('Dataset name is empty')
            return None
        if name in self._local_dataset:
            return [i for i in load_sklearn_toys()][self._local_dataset.index(name)]
        else:
            print(name + ' Local Dataset not found')
            return None

    @property
    def experiments(self):
        return self._experiments


def main():
    """
    Get the dataset from load_sklearn_toys
    The dataset index is 3
    """
    from pypadre.app import p_app
    p_app.set_printer(print)
    logger = PadreLogger()
    logger.backend = p_app.repository
    add_logger(logger=logger)
    # NOTE: Server MUST BE RUNNING!!! See Padre Server!
    # Start PADRE Server and run
    ds = None
    try:
        p_app.datasets.list_datasets()
        ds = p_app.datasets.get_dataset("http://localhost:8080/api/datasets/5")
    except:
        ds = [i for i in load_sklearn_toys()][2]

    experiment_helper = experimentHelper()

    # ex.report_results() # last step, but we can also look that up on the server

    # Testing Support Vector Classification using default values
    workflow = experiment_helper.create_test_pipeline(['SVC'])
    experiment_helper.create_experiment(name='Experiment1', description='This is the first test experiment',
                                        dataset=experiment_helper.get_local_dataset('Breast_Cancer'), workflow=workflow)

    # Create another experiment which has a transformation and logistic regression
    workflow = experiment_helper.create_test_pipeline(['pca', 'logistic'])
    experiment_helper.create_experiment(name='Experiment2', description='This is the second test experiment',
                                        dataset=experiment_helper.get_local_dataset('Iris'), workflow=workflow)

    # Experiment using SVD in the pipeline
    # Setting parameters for estimator 'LSA'/Truncated SVD
    params = {'n_components': 5}
    param_value_dict = {'LSA': params}
    workflow = experiment_helper.create_test_pipeline(['LSA'], param_value_dict)
    experiment_helper.create_experiment(name='Experiment3', description='This is the third test experiment',
                                        dataset=experiment_helper.get_local_dataset('Diabetes'), workflow=workflow)

    # Testing isometric mapping with default values
    workflow = experiment_helper.create_test_pipeline((['isomap']))
    experiment_helper.create_experiment(name='Experiment4', description='This is the fourth test experiment',
                                        dataset=experiment_helper.get_local_dataset('Diabetes'), workflow=workflow)

    # Testing setting parameters to isometric mapping
    params = {'n_neighbors': 10, 'n_components': 5}
    param_value_dict = {'isomap': params}
    workflow = experiment_helper.create_test_pipeline(['isomap'], param_value_dict)
    experiment_helper.create_experiment(name='Experiment5',
                                        description='This is the fifth test experiment',
                                        dataset=experiment_helper.get_local_dataset('Boston_House_Prices'),
                                        workflow=workflow)

    # Testing the combination of PCA and Local Linear Embedding
    workflow = experiment_helper.create_test_pipeline(['pca','lle'])
    experiment_helper.create_experiment(name='Experiment6', description='This is the sixth test experiment',
                                        dataset=experiment_helper.get_local_dataset('Diabetes'), workflow=workflow)

    # Testing Support Vector Regression on the Diabetes dataset
    workflow = experiment_helper.create_test_pipeline(['SVR'])
    experiment_helper.create_experiment(name='Experiment7', description='This is the seventh test experiment',
                                        dataset=experiment_helper.get_local_dataset('Diabetes'), workflow=workflow)

    # Testing missing parameters/error cases when initializing the Experiment class
    # The aim of this is that, while running the experiment it should never crash
    # Missing name
    workflow = experiment_helper.create_test_pipeline(['LSA'])
    experiment_helper.create_experiment(name=None, description='This is the third test experiment',
                                        dataset=experiment_helper.get_local_dataset('Diabetes'), workflow=workflow)

    # Missing description
    workflow = experiment_helper.create_test_pipeline(['LSA'])
    experiment_helper.create_experiment(name='Error2', description=None,
                                        dataset=experiment_helper.get_local_dataset('Diabetes'), workflow=workflow)

    # Missing dataset
    workflow = experiment_helper.create_test_pipeline(['LSA'])
    experiment_helper.create_experiment(name='Error3', description='This experiment is missing the dataset',
                                        dataset=experiment_helper.get_local_dataset('Missing_dataset'), workflow=workflow)

    # Missing workflow
    workflow = experiment_helper.create_test_pipeline(['Wrong_classifier'])
    experiment_helper.create_experiment(name='Error4 ', description='This experiment is missing a workflow',
                                        dataset=experiment_helper.get_local_dataset('Diabetes'), workflow=workflow)

    # Missing backend
    # This will not result in an error as backend dependency has been moved from Experiment to the logger
    workflow = experiment_helper.create_test_pipeline(['LSA'])
    experiment_helper.create_experiment(name='Error5 ', description='This experiment is missing a workflow',
                                        dataset=experiment_helper.get_local_dataset('Diabetes'), workflow=workflow)

    # Applying a classifer on a continous dataset
    workflow = experiment_helper.create_test_pipeline(['SVC'])
    experiment_helper.create_experiment(name='Error6', description='This experiment has a classifier/dataset mismatch',
                                        dataset=experiment_helper.get_local_dataset('Boston_House_Prices'), workflow=workflow)

    # Applying a linear regressor and classifier
    # Runtime exception when logistic regression is combined with SVC
    # Currently linear is a estimator that is not present in the estimator components
    workflow = experiment_helper.create_test_pipeline(['logistic', 'SVC'])
    experiment_helper.create_experiment(name='Error7', description='This experiment has a regressor and classifier',
                                        dataset=experiment_helper.get_local_dataset('Iris'), workflow=workflow)

    experiments_dict = experiment_helper.experiments
    # Run all the experiments in the list
    for experiment in experiments_dict:
        ex = Experiment(name=experiment,
                        description=experiments_dict.get(experiment).get('description'),
                        dataset=experiments_dict.get(experiment).get('dataset', None),
                        workflow=experiments_dict.get(experiment).get('workflow', None))
        conf = ex.configuration()  # configuration, which has been automatically extracted from the pipeline
        pprint.pprint(ex.hyperparameters())  # get and print hyperparameters
        ex.run()  # run the experiment and report

    print("========Available experiments=========")
    for idx, ex in enumerate(p_app.experiments.list_experiments()):
        print("%d: %s" % (idx, str(ex)))
        for idx2, run in enumerate(p_app.experiments.list_runs(ex)):
            print("\tRun: %s" % str(run))


if __name__ == '__main__':
    main()
