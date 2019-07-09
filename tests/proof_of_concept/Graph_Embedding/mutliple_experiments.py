import pandas as pd
from pypadre.ds_import import load_pandas_df
from pypadre.experimentcreator import ExperimentCreator
import numpy as np

def split(idx):
    state = np.random.get_state()

    training_size = int(.8 * len(idx))
    np.random.seed(0)
    shuffle_indices = np.random.permutation(np.arange(len(idx)))
    return shuffle_indices[0:training_size], shuffle_indices[training_size:], None


def main():
    from pypadre.app import p_app

    p_app.set_printer(print)
    ## WIKIPEDIA dataset
    path_dataset = 'data/Wikipedia.csv'
    df1 = pd.read_csv(path_dataset, index_col=0)
    ds1 = load_pandas_df(df1, target_features=['2'], name="Wikipedia")
    ## Cora datatset
    path_dataset = 'data/Cora.csv'
    df2 = pd.read_csv(path_dataset, index_col=0)
    ds2 = load_pandas_df(df2, target_features=['2'], name="Cora")
    ## CiteSeer dataset
    path_dataset = 'data/CiteSeer.csv'
    df3 = pd.read_csv(path_dataset, index_col=0)
    ds3 = load_pandas_df(df3, target_features=['2'], name="CiteSeer")

    #Node classification test
    params_dict = dict()
    preprocessing_params_dict = dict()
    experiment_helper = ExperimentCreator()

    param_nc = {'classifier': [{'estimator': {'name': 'logistic regression', 'params': {'penalty': 'l2', 'tol': 0.0001}},
                               'strategy': {'name': 'one vs rest classifier', 'params': {'n_jobs': 1}}},
                               {'estimator': {'name': 'logistic regression',
                                              'params': {'penalty': 'l1', 'tol': 0.0001}},
                                'strategy': {'name': 'one vs rest classifier', 'params': {'n_jobs': 1}}}
                               ]}
    params_dict['node_classification'] = param_nc
    param_emb = {"dataset": "Wikipedia",
                 "dimension": [28,56,128,256],
                 "alpha": [0],
                 "seed": [1],
                 "min_count": [0],
                 "window": [4,5,8],
                 "sg": [1],
                 "hs": [1],
                 "path_length": [40,60,80],
                 "num_paths": [5,10,20]
                    }
    preprocessing_params_dict['deepwalk'] = param_emb
    workflow = experiment_helper.create_test_pipeline(['node_classification'])
    preprocessing_workflow = experiment_helper.create_test_pipeline(['deepwalk'])
    params_dict = experiment_helper.convert_alternate_estimator_names(params_dict)

    experiment_helper.create(name='Grid_search_Node_classification_1',
                             description='multiple classifiers with deepwalk embeddings',
                             dataset_list=[ds1],
                             workflow=workflow,
                             strategy='function',
                             keep_splits=True,
                             keep_attributes=False,
                             preprocessing=preprocessing_workflow,
                             function= split,
                             params=params_dict,
                             preprocessing_params=preprocessing_params_dict)

    #Link Prediction test
    # param_value_dict = dict()
    #
    # param_lp = {"Classifier": {"estimator": {"name": "logistic regression","params": {"C": 1}}},
    #             "Embedding": [{"name": "deepwalk","params": {"dimension": 128,"dataset" : "CiteSeer","num_paths": 10,"path_legnth": 80,"alpha": 0,"seed": 0}},
    #                           {"name": "deepwalk","params": {"dimension": 64,"dataset" : "CiteSeer","num_paths": 10,"path_legnth": 80,"alpha": 0,"seed": 0}}],
    #             "Graph": {"name" : "CiteSeer","params": {"prop_pos" : 0.5,"prop_neg" : 0.5,"is_directed" : False,"workers" : 1,"random_seed" : 0}},
    #             "num_iteration": [2,4]}
    # param_value_dict['lp'] = param_lp
    # workflow = experiment_helper.create_test_pipeline(['lp'])
    # experiment_param_dict['Grid_search_Link_Prediction_1'] = experiment_helper.convert_alternate_estimator_names(param_value_dict)
    # experiment_helper.create(name='Grid_search_Link_Prediction_1',
    #                          description='multiple classifiers with the same embedding',
    #                          dataset_list=[ds],
    #                          workflow=workflow,
    #                          strategy=None,
    #                          params=experiment_param_dict.get('Grid_search_Link_Prediction_1'))

    experiment_helper.execute()

if __name__== "__main__":
    main()

    # param_emb = {'embedding': {'name': 'deepwalk','params': {'alpha': 0,'dataset': 'CiteSeer','dimension': 128,'num_paths': 10,'path_legnth': 80,'seed': 0}}}

    # emb = WrapperEmbeddings(params=param_emb)
    #
    # #multiple classifiers
    # param_classifiers = [{'classifier': {'estimator': {'name': 'logistic regression','params': {}},
    #                      'strategy': {'name': 'one vs rest classifier', 'params': {'n_jobs': 1}}},
    #                      'num_shuffles': 5,
    #                      'splits': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    #                       'embedding': {'name': 'deepwalk','params': {'alpha': 0,'dataset': 'CiteSeer',
    #                                                                   'dimension': 128,'num_paths': 10,
    #                                                                   'path_legnth': 80,'seed': 0}}},
    #                     {'classifier': {'estimator': {'name': 'random forest classifier','params': {}},
    #                      'strategy': {'name': 'one vs rest classifier', 'params': {'n_jobs': 1}}},
    #                      'num_shuffles': 5,
    #                      'splits': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    #                      'embedding': {'name': 'deepwalk','params': {'alpha': 0,'dataset': 'CiteSeer','dimension': 128,
    #                                                                  'num_paths': 10,'path_legnth': 80,'seed': 0}}},
    #                     # {'classifier': {'estimator': {'name': 'SGD classifier','params': {}},
    #                     #  'strategy': {'name': 'one vs rest classifier', 'params': {'n_jobs': 1}}},
    #                     #  'num_shuffles': 5,
    #                     #  'splits': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    #                     #  'embedding': {'name': 'deepwalk','params': {'alpha': 0,'dataset': 'CiteSeer','dimension': 128,
    #                     #                                              'num_paths': 10,'path_legnth': 80,'seed': 0}}},
    #                      {'classifier': {'estimator': {'name': 'k-nn classifier', 'params': {}},
    #                                      'strategy': {'name': 'one vs rest classifier', 'params': {'n_jobs': 1}}},
    #                       'num_shuffles': 5,
    #                       'splits': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    #                       'embedding': {'name': 'deepwalk',
    #                                     'params': {'alpha': 0, 'dataset': 'CiteSeer', 'dimension': 128,
    #                                                'num_paths': 10, 'path_legnth': 80, 'seed': 0}}},
    #                      {'classifier': {'estimator': {'name': 'gaussian naive bayes', 'params': {}},
    #                                      'strategy': {'name': 'one vs rest classifier', 'params': {'n_jobs': 1}}},
    #                       'num_shuffles': 5,
    #                       'splits': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    #                       'embedding': {'name': 'deepwalk',
    #                                     'params': {'alpha': 0, 'dataset': 'CiteSeer', 'dimension': 128,
    #                                                'num_paths': 10, 'path_legnth': 80, 'seed': 0}}}
    #                      ]
    #
    #
    # for param_clf in param_classifiers:
    #     clf = WrapperNodeClassification(params=param_clf)
    #
    #     workflow = Pipeline([('Emb',emb),('nclf',clf)])
    #     ex = Experiment(name="Node classification with %s" % (param_clf.get('classifier').get('estimator').get('name')),
    #                     description="Evaluating graph embedding",
    #                     dataset=ds,
    #                     workflow=workflow,
    #                     strategy=None)
    #     ex.run()