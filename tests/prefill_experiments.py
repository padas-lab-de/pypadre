"""
This file aims to create multiple different experiments to prefill the server.
The experiments work on the sklearn toy datasets and are a combination of classification and regression tasks
Some experiments will also contain a grid search(testing multiple hyperparameters)
"""
from pypadre.pod.experimentexecutor import ExperimentExecutor
from pypadre.core.events import trigger_event
from pypadre.app import p_app

def main():
    trigger_event('EVENT_LOG_EVENT', source='Experiment_Executor_Test',
                  message='Experiment Executor Starting')
    '''
    CLASSIFICATION EXPERIMENTS
    '''

    # A decision tree classifier with Iris and Digits datasets and multiple parameter assignments
    workflow = p_app.experiment_creator.create_test_pipeline(['decision tree classifier'])
    params = {'max_depth_tree': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'min_samples_split': [2, 3, 4, 5, 6, 7]}
    param_value_dict = {'decision tree classifier': params}
    p_app.experiment_creator.create(name='Decision Tree',
                                    description='This is an experiment with the decision tree classifier',
                                    dataset_list=['Iris', 'Digits'],
                                    workflow=workflow,
                                    params=param_value_dict)

    # A K Nearest Neighbors Classifier using a single dataset
    workflow = p_app.experiment_creator.create_test_pipeline(['k-nn classifier'])
    params = {'n_neighbors': [1, 4, 5],
              'distance_metric': ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']}
    param_value_dict = {'k-nn classifier': params}
    p_app.experiment_creator.create(name='KNN',
                                    description='This is an experiment using the KNN classifier',
                                    dataset_list=['Iris'],
                                    workflow=workflow,
                                    params=param_value_dict)

    # SVM based classification experiment
    workflow = p_app.experiment_creator.create_test_pipeline(['SVC'])
    params = {'probability': True}
    param_value_dict = {'SVC': params}
    p_app.experiment_creator.create(name='SVC',
                                    description='This is an experiment with no parameters',
                                    dataset_list=['Iris', 'Digits'],
                                    workflow=workflow,
                                    params=param_value_dict)

    # Combination of multiple estimators in an experiment
    workflow = p_app.experiment_creator.create_test_pipeline(['PCA', 'random forest classifier'])
    p_app.experiment_creator.create(name='Random Forest Classifier with PCA',
                                    description=' This is an experiment that has a PCA combined with a Random Forest',
                                    dataset_list=['Digits'],
                                    strategy='cv',
                                    workflow=workflow)
    
    # Combination of multiple experiments
    workflow = p_app.experiment_creator.create_test_pipeline(['PCA', 'random forest classifier'])
    params_pca = {'num_components': [2, 3, 4]}
    params_rf = {'num_estimators': [3, 5, 7, 9, 11],
                 'max_depth_tree': [3, 4, 5, 6, 7, 8]}
    param_value_dict = {'random forest classifier': params_rf, 'pca': params_pca}
    p_app.experiment_creator.create(name='Grid search with Random Forest Classifier and PCA',
                                    description=' This is an experiment that has a PCA combined with a Random Forest',
                                    dataset_list=['Digits', 'Iris'],
                                    strategy='cv',
                                    workflow=workflow,
                                    params=param_value_dict)

    workflow = p_app.experiment_creator.create_test_pipeline(['bagging classifier'])
    p_app.experiment_creator.create(name='Experiment using bagging classifier',
                                    description='A sample experiment using the bagging classifier',
                                    dataset_list=['Iris'],
                                    strategy='random',
                                    workflow=workflow)

    '''
    REGRESSION EXPEIRIMENTS
    '''

    # Support vector regression with multiple dataset including a classification dataset
    params_pca = {'num_components': [2, 3, 4, 5, 6]}
    params_svr = {'C': [0.5, 1.0, 1.5],
                  'poly_degree': [1, 2, 3]}
    params_dict = {'SVR': params_svr, 'pca': params_pca}
    workflow = p_app.experiment_creator.create_test_pipeline(['pca', 'SVR'])
    params_dict = p_app.experiment_creator.convert_alternate_estimator_names(params_dict)
    p_app.experiment_creator.create(name='Regression using PCA and SVR',
                                    description='Grid search experiment with SVR',
                                    dataset_list=['Boston_House_Prices', 'Diabetes', 'Digits'],
                                    workflow=workflow,
                                    params=params_dict
                                    )

    workflow = p_app.experiment_creator.create_test_pipeline(['linearSVR'])
    p_app.experiment_creator.create(name='Linear SVR',
                                    description='Experiment using linear SVR and diabetes dataset',
                                    workflow=workflow,
                                    dataset_list='Diabetes')

    # Experiment using KNN regressor
    params = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
    params_dict = {'k-nearest neighbors regressor': params}
    workflow = p_app.experiment_creator.create_test_pipeline(['k-nearest neighbors regressor'])
    p_app.experiment_creator.create(name='KNN Regressor',
                                    description='Experiment using KNN Regressor',
                                    workflow=workflow,
                                    dataset_list='Boston_House_Prices',
                                    params=params_dict)

    workflow = p_app.experiment_creator.create_test_pipeline(['GPR'])
    p_app.experiment_creator.create(name='Gaussian Process Regression',
                                    description='Experiment with Gaussian Process Regression on the Diabetes dataset',
                                    workflow=workflow,
                                    dataset_list='Diabetes')

    workflow = p_app.experiment_creator.create_test_pipeline(['PLS Regession'])
    params = {'num_components': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
    params_dict = {'PLS Regession': params}
    p_app.experiment_creator.create(name='PLS Regression',
                                    description='Experiment with PLS based regression',
                                    workflow=workflow,
                                    dataset_list=['Boston_House_Prices', 'Diabetes'],
                                    params=params_dict)

    experiments_list = p_app.experiment_creator.createExperimentList()
    experiments_executor = ExperimentExecutor(experiments=experiments_list)
    import time
    c1 = time.time()
    experiments_executor.execute(local_run=True, threads=1)
    c2 = time.time()
    print('Execution time:{time_diff}'.format(time_diff=c2 - c1))


if __name__ == '__main__':
    main()
