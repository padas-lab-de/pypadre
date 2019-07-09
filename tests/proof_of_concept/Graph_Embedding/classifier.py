from pypadre.core.wrappers.wrapper_graphembeddings import *
from sklearn.pipeline import Pipeline
from pypadre.core import Experiment
from pypadre.ds_import import *
import pprint
import numpy as np

def split(idx):
    state = np.random.get_state()

    training_size = int(.8 * len(idx))
    np.random.seed(0)
    shuffle_indices = np.random.permutation(np.arange(len(idx)))
    return shuffle_indices[0:training_size], shuffle_indices[training_size:], None

def classifier():

    # Load dataset
    from pypadre.app import p_app
    p_app.set_printer(print)

    ## WIKIPEDIA dataset
    path_dataset = 'data/Wikipedia.csv'
    df1 = pd.read_csv(path_dataset, index_col=0)
    ds1 = load_pandas_df(df1, target_features=['2'],name="Wikipedia")
    ## Cora datatset
    path_dataset = 'data/Cora.csv'
    df2 = pd.read_csv(path_dataset, index_col=0)
    ds2 = load_pandas_df(df2, target_features=['2'], name="Cora")
    ## CiteSeer dataset
    path_dataset = 'data/CiteSeer.csv'
    df3 = pd.read_csv(path_dataset, index_col=0)
    ds3 = load_pandas_df(df3, target_features=['2'], name="CiteSeer")

    #TODO add struc2vec and graphrep
    import json
    with open('deepwalk_nc.json') as json_data:
        params_emb = json.load(json_data)
    with open('config.json') as json_data:
        params = json.load(json_data)
    emb = WrapperDeepwalk(params=params_emb)
    clf = WrapperNodeClassification(params=params)
    transformers = [('deepwalk',emb)]
    preprocessing_workflow = Pipeline(transformers)
    estimators = [('nclf', clf)]
    workflow = Pipeline(estimators)
    #fn=splitting_fn , strategy='function', no_shuffle=False
    ex = Experiment(name="Node_classification(Cora)",
                    description="Evaluating graph embeddings used on Cora dataset using node classification",
                    dataset= ds2,
                    workflow=workflow,
                    preprocessing= preprocessing_workflow,
                    keep_attributes = False,
                    keep_splits=True,
                    strategy = 'function',
                    function= split)
    conf = ex.configuration()  # configuration, which has been automatically extracted from the pipeline
    pprint.pprint(ex.hyperparameters())  # get and print hyperparameters
    ex.execute()
    # p_app.authenticate("Weissger","test")
    # p_app.experiments.upload_local_experiment("Node_classification(CiteSeer)")

if __name__ == "__main__":


    classifier()