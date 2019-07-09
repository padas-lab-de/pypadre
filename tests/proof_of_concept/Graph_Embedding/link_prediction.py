from pypadre.core.wrappers.wrapper_graphembeddings import WrapperLinkPrediction
from sklearn.pipeline import  Pipeline
from pypadre.core import Experiment
from pypadre.ds_import import *



def classifier():
    # Load dataset
    # edge_pairs = []
    #
    # fl = open('CiteSeer.edges', 'r')
    # for line in fl:
    #     a = line.strip('\n')
    #     edge_pairs.append(a)
    #
    # fl.close()
    path_dataset = 'CiteSeer.csv'
    df = pd.read_csv(path_dataset,index_col=0)
    ds = load_pandas_df(df, ['2'])

    import json
    with open('config_linkprediction.json') as json_data:
        params = json.load(json_data)
    clf = WrapperLinkPrediction(params=params)

    # estimators = [('emb', emb),('clf', OneVsRestClassifier(LogisticRegression()))]
    estimators = [('lp', clf)]
    workflow = Pipeline(estimators)
    #fn=splitting_fn , strategy='function', no_shuffle=False
    ex = Experiment(name="Link_Prediction",
                    description="Evaluating graph embedding",
                    dataset= ds,
                    workflow=workflow,
                    strategy = None)
    ex.run()

if __name__ == "__main__":


    classifier()