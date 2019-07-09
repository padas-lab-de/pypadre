import numpy as np
import json
import importlib
import os
import copy
from scipy.sparse import csc_matrix
import networkx as nx
import random
import pickle
from collections import defaultdict
import hashlib
__version__ = '0.0.0'
__doc__ = "This script implements the wrapper function of the graph embeddings techniques evaluation to be used then in a Scikit-learn pipeline"


class WrapperNodeClassification:
    __doc__ = "This script implements the wrapper function of the graph embeddings techniques evaluation with node classification to be used then in a Scikit-learn pipeline"

    # The different dictionaries containing information about the objects, and its parameters etc
    algorithms_dict = None


    # Classifier related variables
    strategy = None
    estimator = None
    metric = None

    #For the hyperparameters.json file
    classifier_params = dict()

    params = dict()

    _id = None

    probability = False
    #for results
    all_results = defaultdict(list)

    def __init__(self, params=None):
        """
        The initialization function for the node classification wrapper
        :param params:  the parameters for creating the node classification model
        e.g: params = {"classifier_params": {"strategy": {"name": "one vs rest classifier","params": {"n_jobs": 1}},
        "estimator": {"name": "logistic regression","params": {"penalty": "l2","tol": 0.0001}}}}
        """
        default_params = dict()
        default_params['classifier_params'] = {'estimator': {'name': 'logistic regression', 'params': {}},
                                                'strategy': {'name': 'one vs rest classifier', 'params': {}}}
        if params is None:
            params = copy.deepcopy(default_params)
        if not self.load_components():
            return

        self.params = copy.deepcopy(params)


        #Get the parameters of the classifier
        classifier = params.get('classifier_params',dict())
        strategy = classifier.get('strategy', dict())
        estimator = classifier.get('estimator' , dict())
        strategy_name = strategy.get('name', 'one vs rest classifier')
        estimator_name = estimator.get('name', 'logistic regression')
        estimator_params = estimator.get('params', None)
        strategy_params = strategy.get('params' , None)
        self.classifier = self.create_estimator(strategy_name=strategy_name,estimator_name=estimator_name,
                                                strat_params=strategy_params,estim_params=estimator_params)
        self.classifier_params = copy.deepcopy(self.classifier.get_params())
        self.classifier_params['name'] = self.classifier.__repr__()[:self.classifier.__repr__().find('(')]
        #a temporary fix for a bug
        self.classifier_params['estimator'] = estimator_name

        #Probablity attribute
        if getattr(self.classifier,"predict_proba", None):
            self.probability = True

        #save parameters #TODO digest
        hashing_string = str(self.classifier_params)
        self._id = hashlib.sha1(hashing_string.encode()).hexdigest()
        self.params['run_id'] = self._id





    def set_params(self, **params):
        """
        This function allows individual parameters to be set to our classifier
        :param params: the parameters to be set
        :return: None
        """
        if not self.load_components():
            return

        for param in params:
            self.params[param] = params.get(param)

        #Classifier
        if params.get('classifier_params',None) is not None:
            classifier = params.get('classifier_params', dict())
            strategy = classifier.get('strategy', dict())
            estimator = classifier.get('estimator', dict())
            strategy_name = strategy.get('name', 'one vs rest classifier')
            estimator_name = estimator.get('name', 'logistic regression')
            estimator_params = estimator.get('params', None)
            strategy_params = strategy.get('params', None)
            self.classifier = self.create_estimator(strategy_name=strategy_name, estimator_name=estimator_name,
                                                    strat_params=strategy_params, estim_params=estimator_params)
            self.classifier_params = copy.deepcopy(self.classifier.get_params())
            self.classifier_params['name'] = self.classifier.__repr__()[:self.classifier.__repr__().find('(')]
            # a temporary fix for a bug
            self.classifier_params['estimator'] = estimator_name

            if self._id is None:
                hashing_string = str(self.classifier_params)
                self._id = hashlib.sha1(hashing_string.encode()).hexdigest()
                self.params['run_id'] = self._id

    def get_params(self):
        """
        A function that return the hyperparameters of the wrapper
        :return: params : dictionary of the parameters
        """
        return self.params

    def fit(self,X,y):
        """
        This function runs the training of the classifier
        :param X: Training embedding features vector
        :param y: Labels corresponding to the feature vector
        :return: None
        """
        self.probabilities = None

        X_, y_ = X, y.reshape(-1,1)

        from sklearn.preprocessing import MultiLabelBinarizer
        self.binarizer = MultiLabelBinarizer(sparse_output=True)
        self.binarizer.fit(y_)
        y_train = self.binarizer.transform(y_)

        # train the model
        self.classifier.fit(X_,y_train)

        self.top_k_list = [len(l) for l in y_] #Used for prediction later on in score or predict

        return self

    def predict(self,X):
        """
        this function tests our fitted classifier
        :param X: Input feature vectors
        :return: predicted labels
        """
        if len(self.top_k_list) != X.shape[0]:
            self.top_k_list = [1 for i in range(X.shape[0])]

        probs = np.asarray(self.classifier.predict_proba(X))
        all_labels = []
        preds = []
        for i, k in enumerate(self.top_k_list):
            probs_ = probs[i, :]
            labels = self.classifier.classes_[probs_.argsort()[-k:]].tolist()
            probs_[:] = 0
            probs_[labels] = 1
            all_labels.append(probs_)
            preds.append(labels)

        self.preds = np.asarray(all_labels) #used in the score function

        return np.asarray(preds).reshape(len(preds))

    def predict_proba(self,X):

        if self.probabilities is None:
            probs = self.classifier.predict_proba(X)
        else:
            probs = self.probabilities

        return probs

    def score(self,X,y):
        """
        Scoring function for micro and macro f1 scores
        :param X: Input features or embeddings
        :param y: Truth values or targets
        :return: macro f1 score of the classifier
        """
        y_= y.reshape(-1,1)
        self.top_k_list = [len(l) for l in y_]

        preds = self.predict(X)

        from sklearn.metrics import f1_score
        # self.binarizer.fit(y_)
        y_test = self.binarizer.transform(y_)
        averages = ["micro" , "macro"]
        scores = {}
        for average in averages:
            scores[average]=f1_score(y_test, self.preds, average=average)
            print("{}_f1-score : {}".format(average,f1_score(y_test, self.preds, average=average)))

        return scores['macro']


    def create_estimator(self, strategy_name='one vs rest classifier', estimator_name='logistic regression'
                         , strat_params=None , estim_params=None):
        """
        This function return an object of the required estimator
        :param strategy_name: Classification strategy
        :param estimator_name: Name of the estimator

        :return: Object of the estimator
        """
        #TODO fix this according to mappings.json
        if strat_params is None and estim_params is None:
            strat_params = dict()
            estim_params = dict()

        estimator = None
        estimator_name = str(estimator_name)
        strategy = None
        strategy_name = str(strategy_name)

        strategy_details = self.algorithms_dict.get(strategy_name, None)
        curr_params_ = None
        if strategy_details is not None:
            path = strategy_details.get('path', None)

            if path is not None:
                curr_params_ = dict()
                strategy_params = strategy_details.get('params', None)

                if strategy_params is None:
                    strategy_params = dict()

                for param in strategy_params:

                    param_value = strat_params.get(param, None)

                    if param_value is None and strategy_params.get(param).get('optional') is False:
                        curr_params_ = None
                        break
                    else:
                        if param_value is not None:
                            curr_params_[param] = param_value

                split_idx = path.rfind('.')
                import_path = path[:split_idx]
                class_name = path[split_idx+1:]
                module = importlib.import_module(import_path)
                class_ = getattr(module, class_name)
                strategy = class_

        estimator_details = self.algorithms_dict.get(estimator_name, None)
        if estimator_details is not None:
            path = estimator_details.get('path',None)

            if path is not None:
                curr_params = dict()

                estimator_params = estimator_details.get('params',None)
                if estimator_params is  None:
                    estimator_params = dict()

                for param in estimator_params:

                    param_value = estim_params.get(param,None)

                    if param_value is None and estimator_params.get(param).get('optional') is False:
                        curr_params = None
                        break
                    else:
                        if param_value is not None:
                            curr_params[param] = param_value

                split_idx = path.rfind('.')
                import_path = path[:split_idx]
                class_name = path[split_idx + 1:]
                module = importlib.import_module(import_path)
                class_ = getattr(module, class_name)
                estimator = strategy(class_(**curr_params), **curr_params_)

        return estimator

    def load_components(self):
        """
        The function loads different components required for the wrapper such embeddings, estimators
        :return: Boolean indicating the success of the function
        """
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../res/mapping"))
        with open(path + '/mapping.json') as f:
            framework_dict = json.load(f)
        algorithms = framework_dict.get('algorithms', None)
        self.algorithms_dict = dict()
        for alg in algorithms:
            self.algorithms_dict[alg['name']] = dict()
            alg_dict = dict()
            alg_dict['path'] = alg["implementation"]['scikit-learn']
            alg_params = dict()
            model_parameters = alg['hyper_parameters'].get('model_parameters')
            optimisation_params = alg['hyper_parameters'].get('optimisation_parameters')
            for param in model_parameters:
                # alg_params[param['name']] = dict()
                params_dict = dict()
                params_dict['optional'] = param['optional']
                params_dict['default'] = param['scikit-learn']['default_value']
                alg_params[param['scikit-learn']['path']] = copy.deepcopy(params_dict)
            for param in optimisation_params:
                # alg_params[param['name']] = dict()
                params_dict = dict()
                params_dict['optional'] = param['optional']
                params_dict['default'] = param['scikit-learn']['default_value']
                alg_params[param['scikit-learn']['path']] = copy.deepcopy(params_dict)
            alg_dict['params'] = copy.deepcopy(alg_params)
            self.algorithms_dict[alg['name']] = copy.deepcopy(alg_dict)

        if self.algorithms_dict == dict():
            return False
        return True

class WrapperStruct2Vec:
    __doc__ = "This script implements the function of the graph embedding method 'struct2vec' which is a transformer component in sklearn pipeline." \
              "Reference : struc2vec: Learning node representations from structural identity (https://arxiv.org/pdf/1704.03165.pdf)."

    algorithms_dict = None

    _id = None
    model_params = dict()

    params = dict()

    def __init_(self, params:None):
        """
        The initialization function for the struct2vec wrapper
        :param params: parameters dictionary for the embedding techniques
        e.g : params: {}
        """



class WrapperDeepwalk:
    __doc__ = "This script implements the function of the graph embedding method 'deepwalk' which is a transformer component in sklearn pipeline." \
              "Reference : DeepWalk: Online Learning of Social Representations (https://arxiv.org/pdf/1403.6652v2.pdf)."
    algorithms_dict = None

    _id = None
    _set_params = False
    model_params = dict()

    params = dict()

    _reset_id = False

    def __init__(self,params=None):
        """
                The initialization function for the deepwalk wrapper
                :param params: the parameters for applying an embedding to the graphs

                e.g : params = {'dataset':"CiteSeer",'dimension':128,'alpha':0,'seed':1,'min_count':0,'window':5,
                                    'sg':1,'hs':1,'path_length':80,'num_paths':10}
                """

        default_params = {'dataset':"CiteSeer",'dimension':128,'alpha':0,'seed':1,'min_count':0,'window':5,
                                    'sg':1,'hs':1,'path_length':80,'num_paths':10}
        if params is None:
            params = copy.deepcopy(default_params)
            if not self.load_components():
                return

        self.params = copy.deepcopy(params)

        self.embedding_name = "deepwalk"
        self.dataset = self.params.get('dataset','CiteSeer')
        self.path_length = self.params.get('path_length', 80)
        self.seed = self.params.get('seed', 0)
        self.rand = random.Random(self.seed)
        self.alpha = self.params.get('alpha', 0)
        self.num_paths = self.params.get('num_paths', 10)
        self.dim = self.params.get('dimension', 128)
        self.min_count = self.params.get('min_count', 0)
        self.hs = self.params.get('hs', 1)
        self.sg = self.params.get('gs', 1)
        self.window = self.params.get('window', 5)

        #A flag to change the id when hyperparameters are changed
        self._reset_id = False

        #ID generation for caching/loading the embeddings
        hashing_string = str(self.params)
        self._id = hashlib.sha1(hashing_string.encode()).hexdigest()
        self.params['run_id'] = self._id

    def set_params(self,**params):
        """

        :param params:
        :return:
                """

        for param in params:
            self.params[param] = params.get(param)


        if params.get('dataset', None) is not None:
            self.dataset = self.params.get('dataset','CiteSeer')
        if params.get('path_length', None) is not None:
            self.path_length = self.params.get('path_length',80)
        if params.get('seed', None) is not None:
            self.seed = self.params.get('seed',0)
            self.rand = random.Random(self.seed)
        if params.get('alpha', None) is not None:
            self.alpha = self.params.get('alpha',0)
        if params.get('num_paths', None) is not None:
            self.num_paths = self.params.get('num_paths','CiteSeer')
        if params.get('dimension', None) is not None:
            self.dim = self.params.get('dimension',128)
        if params.get('min_count', None) is not None:
            self.min_count = self.params.get('min_count',0)
        if params.get('hs', None) is not None:
            self.hs = self.params.get('hs',1)
        if params.get('sg', None) is not None:
            self.sg = self.params.get('sg',1)
        if params.get('window', None) is not None:
            self.window = self.params.get('window',1)

        self._reset_id = True



    def get_params(self):
        """
        A function that return the hyperparameters of the wrapper
        :return: params : dictionary of the parameters
        """

        return self.params

    def create_graph(self, edge_list):
        """
        Read the input edge list into a networkx graph
        :param edge_list: array of node pairs (string format)
        :param graph_params: dict of graph parameters
        :return: a graph object
        """
        edge_list_ = np.hstack(edge_list)
        is_directed = False
        G = nx.parse_edgelist(edge_list_, nodetype=int,create_using=nx.DiGraph(),data=[('weight',int)])
        # for edge in G.edges():
        #     G.adj[edge[0]][edge[1]]['weight']=1

        print("Read graph, nodes: %d, edges: %d" % (G.number_of_nodes(), G.number_of_edges()))

        if not is_directed:
            G = G.to_undirected()

        return G

    def random_walk(self, G, path_length, rand=random.Random(0),alpha=0, start=None):

        if start:
            path = [start]
        else:
            path = [rand.choice(list(G.nodes()))]

        while len(path) < path_length:

            cur = path[-1]
            neighbors = list(G.neighbors(cur))
            if len(neighbors) > 0:

                if rand.random() >= alpha:
                    path.append(rand.choice(neighbors))

                else:

                    path.append(path[0])

            else:
                break
        path = [str(node) for node in path]
        return path

    def create_deepwalk_corpus(self,G, num_paths, path_length, alpha=0, rand=random.Random(0)):
        walks = []
        nodes = list(G.nodes())

        print('number of nodes:', len(nodes))

        for cnt in range(num_paths):
            rand.shuffle(nodes)

            print(str(cnt + 1), '/', str(num_paths))
            for node in nodes:
                walks.append(self.random_walk(G,path_length, rand=rand, alpha=alpha, start=node))  # the same code as deepWalk

        return walks

    def deepwalk(self,G):

        if self._id is None or self._reset_id:
            hashing_string = str(self.params)
            self._id = hashlib.sha1(hashing_string.encode()).hexdigest()
            self.params['run_id'] = self._id
            self._reset_id = False
        from gensim.models import Word2Vec, KeyedVectors
        if os.path.isfile(os.path.abspath(
                os.path.join(os.path.dirname(__file__),
                             "../../../tests/proof_of_concept/Graph_Embedding/cached_embeddings/%s_%s.emb" % (
                                     self.embedding_name,self._id)))):
            model = KeyedVectors.load_word2vec_format(os.path.abspath(
                os.path.join(os.path.dirname(__file__),
                             "../../../tests/proof_of_concept/Graph_Embedding/cached_embeddings/%s_%s.emb" % (
                                     self.embedding_name,self._id))), binary=False)
        else:
            walks = self.create_deepwalk_corpus(G, num_paths=self.num_paths, path_length=self.path_length, rand=self.rand, alpha=self.alpha)
            model = Word2Vec(walks, size= self.dim, window=self.window, min_count=self.min_count, workers= 5, hs=self.hs, sg=self.sg)
            model.wv.save_word2vec_format(os.path.abspath(
                os.path.join(os.path.dirname(__file__),
                             "../../../tests/proof_of_concept/Graph_Embedding/cached_embeddings/%s_%s.emb" % (
                                     self.embedding_name,self._id))))

        return model

    def fit(self, X, y=None):
        """

        :param X:
        :param y:
        :return:
        """
        return self

    def transform(self,X):

        # create the graph from the edge pairs in X and create the embeddings
        edge_list = [' '.join(str(i) for i in x) for x in X if x[1] != -1]
        G = self.create_graph(edge_list)
        model = self.deepwalk(G)

        # Extract the nodes id and their labels for the evaluation
        import pandas as pd
        df = pd.DataFrame(X)
        df = df.drop_duplicates(subset=[0]).sort_values(0)
        X_ = df[0].values

        fm = []
        for node_id in X_:
            fm.append(model[str(node_id)])
        Xt = np.asarray(fm)

        return Xt

    def fit_transform(self, X, y):
        #TODO reshape X, get src and target create embeddings and transform

        #create the graph from the edge pairs in X and create the embeddings
        edge_list = [' '.join(str(i) for i in x) for x in X if x[1]!=-1]
        G = self.create_graph(edge_list)
        model = self.deepwalk(G)

        #Extract the nodes id and their labels for the evaluation
        import pandas as pd
        df = pd.DataFrame(X)
        df[2] = y
        df = df.drop_duplicates(subset=[0]).sort_values(0)
        X_ = df[0].values
        y_ = df[2].values

        fm = []
        for node_id in X_:
            fm.append(model[str(node_id)])
        Xt_ = np.asarray(fm)
        columns = [str(i) for i in range(Xt_.shape[1] + 1)]
        # Xt_ = np.append(Xt_, y_.reshape(-1, 1), axis=1)
        Xt = pd.DataFrame(Xt_, columns=columns[:-1])
        Xt[columns[-1]] = y_.reshape(-1, 1)

        return Xt

    def load_components(self):
        """
        The function loads different components required for the wrapper such embeddings, estimators
        :return: Boolean indicating the success of the function
        """
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../res/mapping"))
        with open(path+'/mapping.json') as f:
            framework_dict = json.load(f)
        algorithms = framework_dict.get('algorithms', None)
        self.algorithms_dict = dict()
        for alg in algorithms:
            self.algorithms_dict[alg['name']] = dict()
            alg_dict = dict()
            alg_dict['path'] = alg["implementation"]['scikit-learn']
            alg_params = dict()
            model_parameters = alg['hyper_parameters'].get('model_parameters')
            optimisation_params = alg['hyper_parameters'].get('optimisation_parameters')
            for param in model_parameters:
                # alg_params[param['name']] = dict()
                params_dict = dict()
                params_dict['optional'] = param['optional']
                params_dict['default'] = param['scikit-learn']['default_value']
                alg_params[param['scikit-learn']['path']] = copy.deepcopy(params_dict)
            for param in optimisation_params:
                # alg_params[param['name']] = dict()
                params_dict = dict()
                params_dict['optional'] = param['optional']
                params_dict['default'] = param['scikit-learn']['default_value']
                alg_params[param['scikit-learn']['path']] = copy.deepcopy(params_dict)
            alg_dict['params'] = copy.deepcopy(alg_params)
            self.algorithms_dict[alg['name']] = copy.deepcopy(alg_dict)

        if self.algorithms_dict == dict():
            return False
        return True


class WrapperNode2Vec:
    __doc__ = "This script implements the function of the graph embedding method 'node2vec' which is a transformer component in sklearn pipeline. " \
              "Reference : node2vec: Scalable Feature Learning for Networks (https://arxiv.org/pdf/1607.00653v1.pdf)."
    algorithms_dict = None

    _id = None
    model_params = dict()

    params = dict()

    def __init__(self,params=None):
        """
                The initialization function for the deepwalk wrapper
                :param params: the parameters for applying an embedding to the graphs
                e.g : params = {'dataset':"CiteSeer",'dimension':128,'alpha':0,'seed':1,'min_count':0,'window':5,
                                    'sg':1,'hs':1,'path_length':80,'num_paths':10,'p':1,'q':1,'workers':4,'directed':False}
                """

        default_params = {'dataset':"CiteSeer",'dimension':128,'alpha':0,'seed':1,'min_count':0,'window':5,
                                    'sg':1,'hs':1,'path_length':80,'num_paths':10,'p':1,'q':1,'workers':4,'directed':False}
        if params is None:
            params = copy.deepcopy(default_params)
            if not self.load_components():
                return

        self.params = copy.deepcopy(params)
        self.embedding_name = "node2vec"
        self.dataset = self.params.get('dataset', 'CiteSeer')
        self.path_length = self.params.get('path_length', 80)
        self.seed = self.params.get('seed', 0)
        self.rand = random.Random(self.seed)
        self.alpha = self.params.get('alpha', 0)
        self.num_paths = self.params.get('num_paths', 10)
        self.dim = self.params.get('dimension', 128)
        self.min_count = self.params.get('min_count', 0)
        self.hs = self.params.get('hs', 1)
        self.sg = self.params.get('sg', 1)
        self.window = self.params.get('window', 5)
        self.p = self.params.get('p',1)
        self.q = self.params.get('q',1)
        self.workers = self.params.get('workers',5)
        self.directed = self.params.get('directed',False)

        hashing_string = str(self.params)
        self._id = hashlib.sha1(hashing_string.encode()).hexdigest()
        self.params['run_id'] = self._id

    def set_params(self,**params):
        """

                :param params:
                :return:
                """

        for param in params:
            self.params[param] = params.get(param)

        if params.get('dataset', None) is not None:
            self.dataset = self.params.get('dataset', 'CiteSeer')
        if params.get('path_length', None) is not None:
            self.path_length = self.params.get('path_length', 80)
        if params.get('seed', None) is not None:
            self.seed = self.params.get('seed', 0)
            self.rand = random.Random(self.seed)
        if params.get('alpha', None) is not None:
            self.alpha = self.params.get('alpha', 0)
        if params.get('num_paths', None) is not None:
            self.num_paths = self.params.get('num_paths', 'CiteSeer')
        if params.get('dimension', None) is not None:
            self.dim = self.params.get('dimension', 128)
        if params.get('min_count', None) is not None:
            self.min_count = self.params.get('min_count', 0)
        if params.get('hs', None) is not None:
            self.hs = self.params.get('hs', 1)
        if params.get('sg', None) is not None:
            self.sg = self.params.get('sg', 1)
        if params.get('window', None) is not None:
            self.window = self.params.get('window', 1)
        if params.get('p', None) is not None:
            self.p = self.params.get('p', 1)
        if params.get('q', None) is not None:
            self.q = self.params.get('q', 1)
        if params.get('workers', None) is not None:
            self.workers = self.params.get('workers', 5)
        if params.get('directed', None) is not None:
            self.directed = self.params.get('directed', False)

        if self._id is None and len(self.params.keys()) > 11:
            hashing_string = str(self.params)
            self._id = hashlib.sha1(hashing_string.encode()).hexdigest()
            self.params['run_id'] = self._id


    def get_params(self):
        """
        A function that return the hyperparameters of the wrapper
        :return: params : dictionary of the parameters
        """

        return self.params

    def create_graph(self, edge_list):
        """
        Read the input edge list into a networkx graph
        :param edge_list: array of node pairs (string format)
        :param graph_params: dict of graph parameters
        :return: a graph object
        """
        edge_list_ = np.hstack(edge_list)
        is_directed = self.directed
        G = nx.parse_edgelist(edge_list_, nodetype=int,create_using=nx.DiGraph(),data=[('weight',int)])


        print("Read graph, nodes: %d, edges: %d" % (G.number_of_nodes(), G.number_of_edges()))

        if not is_directed:
            G = G.to_undirected()

        return G

    def alias_draw(self,J, q):
        '''
        Draw sample from a non-uniform discrete distribution using alias sampling.
        '''
        K = len(J)

        kk = int(np.floor(np.random.rand() * K))
        if np.random.rand() < q[kk]:
            return kk
        else:
            return J[kk]

    def alias_setup(self,probs):
        '''
        Compute utility lists for non-uniform sampling from discrete distributions.
        Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
        for details
        '''
        K = len(probs)
        q = np.zeros(K)
        J = np.zeros(K, dtype=np.int)

        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            q[kk] = K * prob
            if q[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            J[small] = large
            q[large] = q[large] + q[small] - 1.0
            if q[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        return J, q

    def get_alias_edge(self, G,src, dst, p, q):
        '''
        Get the alias edge setup lists for a given edge.
        '''

        unnormalized_probs = []
        for dst_nbr in sorted(G.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(G[dst][dst_nbr].get('weight',1.0) / p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(G[dst][dst_nbr].get('weight',1.0))
            else:
                unnormalized_probs.append(G[dst][dst_nbr].get('weight',1.0) / q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]

        return self.alias_setup(normalized_probs)

    def transition_probs(self,G,p,q):
        '''
        	Preprocessing of transition probabilities for guiding the random walks.
        '''

        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [G[node][nbr].get('weight',1.0) for nbr in sorted(G.neighbors(node))]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = self.alias_setup(normalized_probs)

        alias_edges = {}

        if G.is_directed():
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(G,edge[0], edge[1],p,q)
        else:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(G,edge[0], edge[1],p,q)
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge(G,edge[1], edge[0],p,q)

        return alias_nodes,alias_edges

    def node2vec_walk(self, G, path_length, start_node, alias_nodes,alias_edges):

        alias_nodes,alias_edges = alias_nodes,alias_edges

        walk = [start_node]

        while len(walk) < path_length:
            cur = walk[-1]
            cur_nbrs = sorted(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[self.alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    next = cur_nbrs[self.alias_draw(alias_edges[(prev, cur)][0],alias_edges[(prev, cur)][1])]
                    walk.append(next)
            else:
                break

        return walk

    def simulate_walks(self,G,num_walks, walk_length,alias_nodes,alias_edges):
        '''
        		Repeatedly simulate random walks from each node.
        		'''
        walks = []
        nodes = list(G.nodes())
        print('Walk iteration:')
        for walk_iter in range(num_walks):
            print(str(walk_iter + 1), '/', str(num_walks))
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.node2vec_walk(G,walk_length, node,alias_nodes,alias_edges))

        return walks

    def partition_num(self,num_walks,workers):

        if num_walks % workers == 0:
            return [num_walks//workers]*workers
        else:
            return [num_walks//workers]*workers + [num_walks % workers]

    def create_node2vec_corpus(self,G,num_walks,walk_length,alias_nodes,alias_edges,n_jobs):

        from joblib import Parallel, delayed
        import itertools

        results = Parallel(n_jobs=n_jobs, verbose=0)(delayed(self.simulate_walks)(G,num_walks,walk_length,alias_nodes,alias_edges) for num in self.partition_num(num_walks,n_jobs))

        walks = list(itertools.chain(*results))

        return walks
    def node2vec(self,G):

        from gensim.models import Word2Vec, KeyedVectors
        if os.path.isfile(os.path.abspath(
                os.path.join(os.path.dirname(__file__),
                             "../../../tests/proof_of_concept/Graph_Embedding/cached_embeddings/%s_%s.emb" % (
                                    self.embedding_name,self._id)))):
            model = KeyedVectors.load_word2vec_format(os.path.abspath(
                os.path.join(os.path.dirname(__file__),
                             "../../../tests/proof_of_concept/Graph_Embedding/cached_embeddings/%s_%s.emb" % (
                                  self.embedding_name,self._id))), binary=False)
        else:
            alias_nodes,alias_edges = self.transition_probs(G,p=self.p,q=self.q)

            walks = self.create_node2vec_corpus(G, num_walks=self.num_paths, walk_length=self.path_length, alias_nodes=alias_nodes,
                                                alias_edges=alias_edges,n_jobs=self.workers)
            walks = [list(map(str, walk)) for walk in walks]
            model = Word2Vec(walks, size=self.dim, window=self.window,alpha=self.alpha, min_count=self.min_count, workers=5,  sg=self.sg,hs=self.hs)
            model.wv.save_word2vec_format(os.path.abspath(
                os.path.join(os.path.dirname(__file__),
                             "../../../tests/proof_of_concept/Graph_Embedding/cached_embeddings/%s_%s.emb" % (
                                 self.embedding_name,self._id))))

        return model

    def fit(self, X, y=None):
        """

        :param X:
        :param y:
        :return:
        """
        return self

    def transform(self,X):
        # create the graph from the edge pairs in X and create the embeddings
        edge_list = [' '.join(str(i) for i in x) for x in X if x[1] != -1]
        G = self.create_graph(edge_list)
        model = self.node2vec(G)

        # Extract the nodes id and their labels for the evaluation
        import pandas as pd
        df = pd.DataFrame(X)
        df = df.drop_duplicates(subset=[0]).sort_values(0)
        X_ = df[0].values

        fm = []
        for node_id in X_:
            fm.append(model[str(node_id)])
        Xt = np.asarray(fm)

        return Xt

    def fit_transform(self, X, y):
        #TODO reshape X, get src and target create embeddings and transform

        #create the graph from the edge pairs in X and create the embeddings
        edge_list = [' '.join(str(i) for i in x) for x in X if x[1]!=-1]
        G = self.create_graph(edge_list)
        model = self.node2vec(G)

        #Extract the nodes id and their labels for the evaluation
        import pandas as pd
        df = pd.DataFrame(X)
        df[2] = y
        df = df.drop_duplicates(subset=[0]).sort_values(0)
        X_ = df[0].values
        y_ = df[2].values

        fm = []
        for node_id in X_:
            fm.append(model[str(node_id)])
        Xt_ = np.asarray(fm)
        columns = [str(i) for i in range(Xt_.shape[1] + 1)]
        # Xt_ = np.append(Xt_, y_.reshape(-1, 1), axis=1)
        Xt = pd.DataFrame(Xt_, columns=columns[:-1])
        Xt[columns[-1]] = y_.reshape(-1, 1)

        return Xt

    def load_components(self):
        """
        The function loads different components required for the wrapper such embeddings, estimators
        :return: Boolean indicating the success of the function
        """
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../res/mapping"))
        with open(path+'/mapping.json') as f:
            framework_dict = json.load(f)
        algorithms = framework_dict.get('algorithms', None)
        self.algorithms_dict = dict()
        for alg in algorithms:
            self.algorithms_dict[alg['name']] = dict()
            alg_dict = dict()
            alg_dict['path'] = alg["implementation"]['scikit-learn']
            alg_params = dict()
            model_parameters = alg['hyper_parameters'].get('model_parameters')
            optimisation_params = alg['hyper_parameters'].get('optimisation_parameters')
            for param in model_parameters:
                # alg_params[param['name']] = dict()
                params_dict = dict()
                params_dict['optional'] = param['optional']
                params_dict['default'] = param['scikit-learn']['default_value']
                alg_params[param['scikit-learn']['path']] = copy.deepcopy(params_dict)
            for param in optimisation_params:
                # alg_params[param['name']] = dict()
                params_dict = dict()
                params_dict['optional'] = param['optional']
                params_dict['default'] = param['scikit-learn']['default_value']
                alg_params[param['scikit-learn']['path']] = copy.deepcopy(params_dict)
            alg_dict['params'] = copy.deepcopy(alg_params)
            self.algorithms_dict[alg['name']] = copy.deepcopy(alg_dict)

        if self.algorithms_dict == dict():
            return False
        return True


class WrapperEmbeddings:
    __doc__ = "This script implements the wrapper function of the graph embeddings techniques  to be used then in a Scikit-learn pipeline"
    algorithms_dict = None

    embedding = None
    dimension = None

    model_params = dict()

    params = dict()

    _id = None

    def __init__(self, params=None):
        """
        The initialization function for the embeddings wrapper
        :param params: the parameters for applying an embedding to the graphs
        """
        default_params = dict()
        default_params['model'] = {'name': 'Random', 'params': {'dimension': 128}}
        if params is None:
            params = copy.deepcopy(default_params)
        if not self.load_components():
            return

        self.params = copy.deepcopy(params)

        #Get basic info about the embedding like dimension and techniques

        embedding = params.get('model', dict())
        embedding_name = embedding.get('name' , 'Random')
        embedding_params = embedding.get('params', None)
        self.model_params = copy.deepcopy(embedding_params)
        self.model_params['name'] = embedding_name
        hashing_string = str(self.model_params)
        self._id = hashlib.sha1(hashing_string.encode()).hexdigest()
        self.params['run_id'] = self._id

    def set_params(self, **params):
        """

        :param params:
        :return:
        """

        for param in params:
            self.params[param] = params.get(param)


        if params.get('model_params', None) is not None:
            embedding_model = params.get('model_params',dict())
            embedding = embedding_model.get('model', dict())
            embedding_name = embedding.get('name', 'Random')
            embedding_params = embedding.get('params', None)

            self.model_params = copy.deepcopy(embedding_params)
            self.model_params['name'] = embedding_name
            if self._id is None:
                hashing_string = str(self.model_params)
                self._id = hashlib.sha1(hashing_string.encode()).hexdigest()
                self.params['run_id'] = self._id

    def get_params(self):
        """
        A function that return the hyperparameters of the wrapper
        :return: params : dictionary of the parameters
        """

        return self.params

    def create_graph(self, edge_list):
        """
        Read the input edge list into a networkx graph
        :param edge_list: array of node pairs (string format)
        :param graph_params: dict of graph parameters
        :return: a graph object
        """
        edge_list_ = np.hstack(edge_list)
        is_directed = False
        G = nx.parse_edgelist(edge_list_, nodetype=int,create_using=nx.DiGraph())
        for edge in G.edges():
            G.adj[edge[0]][edge[1]]['weight']=1

        print("Read graph, nodes: %d, edges: %d" % (G.number_of_nodes(), G.number_of_edges()))

        if not is_directed:
            G = G.to_undirected()

        return G

    #Node2Vec
    def alias_draw(self,J, q):
        '''
        Draw sample from a non-uniform discrete distribution using alias sampling.
        '''
        K = len(J)

        kk = int(np.floor(np.random.rand() * K))
        if np.random.rand() < q[kk]:
            return kk
        else:
            return J[kk]

    def alias_setup(self,probs):
        '''
        Compute utility lists for non-uniform sampling from discrete distributions.
        Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
        for details
        '''
        K = len(probs)
        q = np.zeros(K)
        J = np.zeros(K, dtype=np.int)

        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            q[kk] = K * prob
            if q[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            J[small] = large
            q[large] = q[large] + q[small] - 1.0
            if q[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        return J, q

    def get_alias_edge(self, G,src, dst, p, q):
        '''
        Get the alias edge setup lists for a given edge.
        '''

        unnormalized_probs = []
        for dst_nbr in sorted(G.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(G[dst][dst_nbr]['weight'] / p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(G[dst][dst_nbr]['weight'])
            else:
                unnormalized_probs.append(G[dst][dst_nbr]['weight'] / q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]

        return self.alias_setup(normalized_probs)

    def transition_probs(self,G,p,q):
        '''
        	Preprocessing of transition probabilities for guiding the random walks.
        '''

        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = self.alias_setup(normalized_probs)

        alias_edges = {}


        if G.is_directed():
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(G,edge[0], edge[1],p,q)
        else:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(G,edge[0], edge[1],p,q)
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge(G,edge[1], edge[0],p,q)

        return alias_nodes,alias_edges

    def node2vec_walk(self, G, path_length, start_node,p,q):

        alias_nodes,alias_edges = self.transition_probs(G,p,q)

        walk = [start_node]

        while len(walk) < path_length:
            cur = walk[-1]
            cur_nbrs = sorted(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[self.alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    next = cur_nbrs[self.alias_draw(alias_edges[(prev, cur)][0],
                                               alias_edges[(prev, cur)][1])]
                    walk.append(next)
            else:
                break

        return walk

    def create_node2vec_corpus(self,G,num_walks, walk_length,p,q):
        '''
        		Repeatedly simulate random walks from each node.
        		'''
        walks = []
        nodes = list(G.nodes())
        print('Walk iteration:')
        for walk_iter in range(num_walks):
            print(str(walk_iter + 1), '/', str(num_walks))
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.node2vec_walk(G,walk_length, node,p,q))

        return walks

    def node2vec(self,G):
        path_length = self.model_params.get('path_length', 80)
        num_paths = self.model_params.get('num_paths', 10)
        dim = self.model_params.get('dimension', 128)
        window = self.model_params.get('window_size', 10)
        p = self.model_params.get('p', 1)
        q = self.model_params.get('q', 1)
        emb = "deepwalk"
        dataset = self.model_params.get('dataset', 'CiteSeer')

        from gensim.models import Word2Vec, KeyedVectors
        if os.path.isfile(os.path.abspath(
                os.path.join(os.path.dirname(__file__),
                             "../../../tests/proof_of_concept/Graph_Embedding/cached_embeddings/%s.emb" % (
                                    self._id)))):
            model = KeyedVectors.load_word2vec_format(os.path.abspath(
                os.path.join(os.path.dirname(__file__),
                             "../../../tests/proof_of_concept/Graph_Embedding/cached_embeddings/%s.emb" % (
                                  self._id))), binary=False)
        else:
            walks = self.create_node2vec_corpus(G, num_walks=num_paths, walk_length=path_length, p=p, q=q)
            walks = [map(str, walk) for walk in walks]
            model = Word2Vec(walks, size=dim, window=window, min_count=0, workers=5,  sg=0,hs=1)
            model.wv.save_word2vec_format(os.path.abspath(
                os.path.join(os.path.dirname(__file__),
                             "../../../tests/proof_of_concept/Graph_Embedding/cached_embeddings/%s.emb" % (
                                 self._id))))

        return model
    def random_walk(self, G, path_length, rand=random.Random(0),alpha=0, start=None):

        if start:
            path = [start]
        else:
            path = [rand.choice(list(G.nodes()))]

        while len(path) < path_length:

            cur = path[-1]
            neighbors = list(G.neighbors(cur))
            if len(neighbors) > 0:

                if rand.random() >= alpha:
                    path.append(rand.choice(neighbors))

                else:

                    path.append(path[0])

            else:
                break
        path = [str(node) for node in path]
        return path

    def create_deepwalk_corpus(self,G, num_paths, path_length, alpha=0, rand=random.Random(0)):
        walks = []
        nodes = list(G.nodes())

        print('number of nodes:', len(nodes))

        for cnt in range(num_paths):
            rand.shuffle(nodes)

            print(str(cnt + 1), '/', str(num_paths))
            for node in nodes:
                walks.append(self.random_walk(G,path_length, rand=rand, alpha=alpha, start=node))  # the same code as deepWalk

        return walks
    def deepwalk(self,G):
        path_length = self.model_params.get('path_length',80)
        seed = self.model_params.get('seed', 0)
        rand = random.Random(seed)
        alpha = self.model_params.get('alpha', 0)
        num_paths = self.model_params.get('num_paths',10)
        dim = self.model_params.get('dimension', 128)
        emb = "deepwalk"
        dataset = self.model_params.get('dataset', 'CiteSeer')


        from gensim.models import Word2Vec, KeyedVectors
        if os.path.isfile(os.path.abspath(
                os.path.join(os.path.dirname(__file__),
                             "../../../tests/proof_of_concept/Graph_Embedding/cached_embeddings/%s.emb" % (
                                     self._id)))):
            model = KeyedVectors.load_word2vec_format(os.path.abspath(
                os.path.join(os.path.dirname(__file__),
                             "../../../tests/proof_of_concept/Graph_Embedding/cached_embeddings/%s.emb" % (
                                     self._id))), binary=False)
        else:
            walks = self.create_deepwalk_corpus(G, num_paths=num_paths, path_length=path_length, rand=rand, alpha=alpha)
            model = Word2Vec(walks, size= dim, window=10, min_count=0, workers= 5, hs=1, sg=0)
            model.wv.save_word2vec_format(os.path.abspath(
                os.path.join(os.path.dirname(__file__),
                             "../../../tests/proof_of_concept/Graph_Embedding/cached_embeddings/%s.emb" % (
                                     self._id))))

        return model

    def random(self, G):
        # dataset = self.model_params.get('dataset','CiteSeer')
        # emb = 'Random'
        emb_dim = self.model_params.get('dimension', 128)
        if os.path.isfile(os.path.abspath(
                os.path.join(os.path.dirname(__file__),
                             "../../../tests/proof_of_concept/Graph_Embedding/cached_embeddings/%s.emb" % (
                                                        self._id)))):
            with open(os.path.abspath(
                os.path.join(os.path.dirname(__file__),
                             "../../../tests/proof_of_concept/Graph_Embedding/cached_embeddings/%s.emb" % (
                                                         self._id))),'rb') as f:
                model = pickle.load(f)
        else:
            model = {}
            for node in G.nodes():
                vector = np.random.rand(emb_dim)
                model[str(node)] = vector
            with open(os.path.abspath(
                os.path.join(os.path.dirname(__file__),
                             "../../../tests/proof_of_concept/Graph_Embedding/cached_embeddings/%s.emb" % (
                                                         self._id))),'wb') as f:
                pickle.dump(model,f,pickle.HIGHEST_PROTOCOL)
        return model
    def create_embedding(self, G):
        embedding_dict = {'deepwalk': self.deepwalk, 'Random': self.random, 'node2vec':self.node2vec}
        emb = self.model_params.get('name', 'Random')

        return embedding_dict[emb](G)
    def fit(self, X, y=None):
        """

        :param X:
        :param y:
        :return:
        """
        return self

    def transform(self,X):
        import pandas as pd
        # extract the nodes and their labels
        df = pd.DataFrame(X)
        df = df.drop_duplicates(subset=[0]).sort_values(0)
        X_ = df[0].values

        # create the graph from the edge pairs
        edge_list = [' '.join(str(i) for i in x) for x in X if x[1] != -1]
        G = self.create_graph(edge_list)

        model = self.create_embedding(G)

        fm = []
        for node_id in X_:
            fm.append(model[str(node_id)])

        Xt = np.asarray(fm)

        return Xt

    def fit_transform(self, X, y):
        #TODO reshape X, get src and target create embeddings and transform
        import pandas as pd
        #extract the nodes and their labels
        df = pd.DataFrame(X)
        df[2] = y
        df = df.drop_duplicates(subset=[0]).sort_values(0)
        X_ = df[0].values
        y_= df[2].values
        #create the graph from the edge pairs
        edge_list = [' '.join(str(i) for i in x) for x in X if x[1]!=-1]
        G = self.create_graph(edge_list)

        model = self.create_embedding(G)

        fm = []
        for node_id in X_:
            fm.append(model[str(node_id)])

        yt = csc_matrix((np.array([1 for i in range(X_.shape[0])]), (np.array([i for i in range(X_.shape[0])]), y_)))
        #yt becomes inculed in X SEE fit function in the last estimator

        Xt = np.asarray(fm)

        return Xt, yt

    def load_components(self):
        """
        The function loads different components required for the wrapper such embeddings, estimators
        :return: Boolean indicating the success of the function
        """
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../res/mapping"))
        with open(path+'/mapping.json') as f:
            framework_dict = json.load(f)
        algorithms = framework_dict.get('algorithms', None)
        self.algorithms_dict = dict()
        for alg in algorithms:
            self.algorithms_dict[alg['name']] = dict()
            alg_dict = dict()
            alg_dict['path'] = alg["implementation"]['scikit-learn']
            alg_params = dict()
            model_parameters = alg['hyper_parameters'].get('model_parameters')
            optimisation_params = alg['hyper_parameters'].get('optimisation_parameters')
            for param in model_parameters:
                # alg_params[param['name']] = dict()
                params_dict = dict()
                params_dict['optional'] = param['optional']
                params_dict['default'] = param['scikit-learn']['default_value']
                alg_params[param['scikit-learn']['path']] = copy.deepcopy(params_dict)
            for param in optimisation_params:
                # alg_params[param['name']] = dict()
                params_dict = dict()
                params_dict['optional'] = param['optional']
                params_dict['default'] = param['scikit-learn']['default_value']
                alg_params[param['scikit-learn']['path']] = copy.deepcopy(params_dict)
            alg_dict['params'] = copy.deepcopy(alg_params)
            self.algorithms_dict[alg['name']] = copy.deepcopy(alg_dict)

        if self.algorithms_dict == dict():
            return False
        return True


class WrapperLinkPrediction:
    __doc__= "This script implements a wrapper for link prediciton evaluation of graph embeddings to be used in a Scikit-learn pipeline"
    algorithms_dict = None

    embedding = None
    dimension = None

    estimator = None

    classifier_params = dict()

    embedding_params = dict()

    graph_params = dict()

    params = dict()

    num_iteration = None



    _id = None

    def __init__(self, params=None):
        """
        The initialization method for the link prediciton wrapper
        :param params:
        """
        default_params = dict()
        default_params['classifier_params'] = {"estimator": {"name": "logistic regression","params": {}}}
        default_params['embedding_params'] = {'name': 'Random', 'params': {'dimension': 128}}
        default_params['graph_params']= {"name" : "CiteSeer","params": {	"prop_pos" : 0.5,	"prop_neg" : 0.5,	"is_directed" : False,	"workers" : 1,	"random_seed" : 0}}

        if params is None:
            params = copy.deepcopy(default_params)
        if not self.load_components():
            return

        self.params = copy.deepcopy(params)



        self.num_iteration = params.get('num_iteration',2)



        #embedding params
        embedding = params.get('embedding_params', dict())
        embedding_name = embedding.get('name', 'Random')
        embedding_params = embedding.get('params', None)
        self.embedding_params = copy.deepcopy(embedding_params)
        self.embedding_params['name'] = embedding_name

        hashing_string = str(self.embedding_params)
        self._id = hashlib.sha1(hashing_string.encode()).hexdigest()
        self.params['run_id'] = self._id

        #Graph params
        graph = params.get('graph_params',dict())
        graph_name = graph.get('name','CiteSeer')
        graph_params = graph.get('params',None)
        self.graph_params = copy.deepcopy(graph_params)
        self.graph_params['name'] = graph_name
        self.rnd_ = np.random.RandomState(seed=graph_params.get('random_seed',1))

        #classifier
        classifier = params.get('classifier_params', dict())
        estimator = classifier.get('estimator', dict())
        estimator_name = estimator.get('name', 'logistic regression')
        estimator_params = estimator.get('params', None)
        self.classifier = self.create_estimator(estimator_name=estimator_name,estim_params=estimator_params)
        self.classifier_params = copy.deepcopy(self.classifier.get_params())
        self.classifier_params['name'] = self.classifier.__repr__()[:self.classifier.__repr__().find('(')]
        # a temporary fix for a bug
        self.classifier_params['estimator'] = estimator_name



    def set_params(self,**params):

        for param in params:
            self.params[param] = params.get(param)

        if params.get('num_iteration',None) is not None:
            self.num_iteration = params.get('num_iteration', 2)

        #embedding params
        if params.get('embedding_params',None) is not None:
            embedding = params.get('embedding_params',dict())
            embedding_name = embedding.get('name', 'Random')
            embedding_params = embedding.get('params', None)
            self.embedding_params = copy.deepcopy(embedding_params)
            self.embedding_params['name'] = embedding_name
            hashing_string = str(self.embedding_params)
            self._id = hashlib.sha1(hashing_string.encode()).hexdigest()
            self.params['run_id'] = self._id

        if params.get('graph_params',None) is not None:
            graph = params.get('graph_params', dict())
            graph_name = graph.get('name', 'CiteSeer')
            graph_params = graph.get('params', None)
            self.graph_params = copy.deepcopy(graph_params)
            self.graph_params['name'] = graph_name
            self.rnd_ = np.random.RandomState(seed=graph_params.get('random_seed', 1))

        if params.get('classifier_params',None) is not None:
            classifier = params.get('classifier', dict())
            estimator = classifier.get('estimator', dict())
            estimator_name = estimator.get('name', 'logistic regression')
            estimator_params = estimator.get('params', None)
            self.classifier = self.create_estimator(estimator_name=estimator_name, estim_params=estimator_params)
            self.classifier_params = copy.deepcopy(self.classifier.get_params())
            self.classifier_params['name'] = self.classifier.__repr__()[:self.classifier.__repr__().find('(')]

    def get_params(self):
        """

        :return:
        """
        return self.params
    def create_graph(self, edge_list):
        """
        Read the input edge list into a networkx graph
        :param edge_list: array of node pairs (string format)
        :param graph_params: dict of graph parameters
        :return: a graph object
        """
        edge_list_ = [' '.join(str(i) for i in x) for x in edge_list if x[1] != -1]
        # edge_list_ = np.hstack(edge_list)
        is_directed = self.graph_params.get('is_directed',False)
        G = nx.parse_edgelist(edge_list_, nodetype=int,create_using=nx.DiGraph())
        for edge in G.edges():
            G.adj[edge[0]][edge[1]]['weight']=1

        print("Read graph, nodes: %d, edges: %d" % (G.number_of_nodes(), G.number_of_edges()))

        if not is_directed:
            G = G.to_undirected()

        return G

    def generate_pos_neg_links(self,G,prop_pos=0.5,prop_neg=0.5):
        """
        Select random existing edges in the graph to be postive links,
        and random non-edges to be negative links.
        Modify graph by removing the postive links.
        """
        #select randomly n edges (positive samples)
        n_edges = G.number_of_edges()
        n_nodes = G.number_of_nodes()



        npos = int(prop_pos * n_edges)
        nneg = int(prop_neg * n_edges)

        n_neighbors = [len(list(G.neighbors(v))) for v in G.nodes()]

        non_edges = [e for e in nx.non_edges(G)]
        print("Finding %d of %d non-edges" % (nneg, len(non_edges)))
        #select nneg pairs of non edges (negative samples)
        random_idx = self.rnd_.choice(len(non_edges),nneg,replace =False)
        neg_edge_list = [non_edges[ii] for ii in random_idx]

        if len(neg_edge_list) < nneg:
            raise RuntimeWarning("Only %d negative edges found" % (len(neg_edge_list)))

        print("Finding %d positive edges of %d total edges" % (npos, n_edges))
        #Finding positive edges and removing them
        edges = list(G.edges())

        pos_edge_list = []
        n_count = 0
        n_ignored_count = 0
        random_idx = self.rnd_.permutation(n_edges)

        for ie in random_idx.tolist():
            edge = edges[ie]
            data = G[edge[0]][edge[1]]
            G.remove_edge(*edge)

            #check if graph is still connected
            reachable_from_v1 = self.bfs(G,edge[0])
            if edge[1] not in reachable_from_v1:
                G.add_edge(*edge,**data)
                n_ignored_count+=1
            else:
                pos_edge_list.append(edge)
                print("Found: %d    " % (n_count), end="\r")
                n_count+=1

            #stopping criteria
            if n_count >= npos:
                break


        pos_edge_list = pos_edge_list
        neg_edge_list = neg_edge_list[:len(pos_edge_list)]

        print('pos_edge_list', len(pos_edge_list))
        print('neg_edge_list',len(neg_edge_list))

        return G,pos_edge_list,neg_edge_list

    def get_selected_edges(self,pos_edge_list,neg_edge_list):

        edges = pos_edge_list + neg_edge_list
        labels = np.zeros(len(edges))
        labels[:len(pos_edge_list)] = 1

        return edges,labels

    def edges_to_features(self,edge_list, edge_function, emb_size, model):
        """
        Given a list of edge lists and a list of labels, create
        an edge feature array using binary_edge_function and
        create a label array matching the label in the list to all
        edges in the corresponding edge list
        :param edge_function:
            Function of two arguments taking the node features and returning
            an edge feature of given dimension
        :param dimension:
            Size of returned edge feature vector, if None defaults to
            node feature size.
        :param k:
            Partition number. If None use all positive & negative edges
        :return:
            feature_vec (n, dimensions), label_vec (n)
        """

        n_total = len(edge_list)
        feature_vec = np.empty((n_total,emb_size), dtype='f')


        for i in range(n_total):
            v1, v2 = edge_list[i]

            # edge-node features
            emb1 = np.asarray(model[str(v1)])
            emb2 = np.asarray(model[str(v2)])
            # emb1 = np.asarray(model[v1])
            # emb2 = np.asarray(model[v2])

            feature_vec[i] = edge_function(emb1,emb2)

        return feature_vec

    def bfs(self,G, source):
        """A fast BFS node generator taken from nx.connected._plain_bfs implementation """
        G_adj = G.adj
        seen = set()
        nextlevel = {source}
        while nextlevel:
            thislevel = nextlevel
            nextlevel = set()
            for v in thislevel:
                if v not in seen:
                    yield v
                    seen.add(v)
                    nextlevel.update(G_adj[v])

    def create_estimator(self, estimator_name='logistic regression', estim_params=None):
        """
        This function return an object of the required estimator
        :param strategy_name: Classification strategy
        :param estimator_name: Name of the estimator

        :return: Object of the estimator
        """

        if  estim_params is None:
            estim_params = dict()

        estimator = None
        estimator_name = str(estimator_name)

        estimator_details = self.algorithms_dict.get(estimator_name, None)
        if estimator_details is not None:
            path = estimator_details.get('path', None)

            if path is not None:
                curr_params = dict()

                estimator_params = estimator_details.get('params', None)
                if estimator_params is None:
                    estimator_params = dict()

                for param in estimator_params:

                    param_value = estim_params.get(param, None)

                    if param_value is None and estimator_params.get(param).get('optional') is False:
                        curr_params = None
                        break
                    else:
                        if param_value is not None:
                            curr_params[param] = param_value

                split_idx = path.rfind('.')
                import_path = path[:split_idx]
                class_name = path[split_idx + 1:]
                module = importlib.import_module(import_path)
                class_ = getattr(module, class_name)
                estimator = class_(**curr_params)

        return estimator

    def random_walk(self, G, path_length, rand=random.Random(0),alpha=0, start=None):

        if start:
            path = [start]
        else:
            path = [rand.choice(list(G.nodes()))]

        while len(path) < path_length:

            cur = path[-1]
            neighbors = list(G.neighbors(cur))
            if len(neighbors) > 0:

                if rand.random() >= alpha:
                    path.append(rand.choice(neighbors))

                else:

                    path.append(path[0])

            else:
                break
        path = [str(node) for node in path]
        return path

    def create_deepwalk_corpus(self,G, num_paths, path_length, alpha=0, rand=random.Random(0)):
        walks = []
        nodes = list(G.nodes())

        print('number of nodes:', len(nodes))

        for cnt in range(num_paths):
            rand.shuffle(nodes)

            print(str(cnt + 1), '/', str(num_paths))
            for node in nodes:
                walks.append(self.random_walk(G,path_length, rand=rand, alpha=alpha, start=node))  # the same code as deepWalk

        return walks
    def deepwalk(self,G):
        path_length = self.embedding_params.get('path_length', 10)
        seed = self.embedding_params.get('seed', 0)
        rand = random.Random(seed)
        alpha = self.embedding_params.get('alpha', 0)
        num_paths = self.embedding_params.get('num_paths', 10)
        dim = self.embedding_params.get('dimension', 128)
        emb = "deepwalk"
        dataset = self.embedding_params.get('dataset', 'CiteSeer')

        from gensim.models import Word2Vec, KeyedVectors
        if os.path.isfile(os.path.abspath(
                os.path.join(os.path.dirname(__file__),
                             "../../../tests/proof_of_concept/Graph_Embedding/cached_embeddings/%s_%s.emb" % (
                                     emb,self._id)))):
            model = KeyedVectors.load_word2vec_format(os.path.abspath(
                os.path.join(os.path.dirname(__file__),
                             "../../../tests/proof_of_concept/Graph_Embedding/cached_embeddings/%s_%s.emb" % (
                                 emb,self._id))), binary=False)
        else:
            walks = self.create_deepwalk_corpus(G, num_paths=num_paths, path_length=path_length, rand=rand, alpha=alpha)
            model = Word2Vec(walks, size=dim, window=10, min_count=0, workers=5, hs=1, sg=0)
            model.wv.save_word2vec_format(os.path.abspath(
                os.path.join(os.path.dirname(__file__),
                             "../../../tests/proof_of_concept/Graph_Embedding/cached_embeddings/%s_%s.emb" % (
                                 emb,self._id))))

        return model

    def random(self, G):
        dataset = self.embedding_params.get('dataset', 'CiteSeer')
        emb = 'Random'
        emb_dim = self.embedding_params.get('dimension', 128)
        if os.path.isfile(os.path.abspath(
                os.path.join(os.path.dirname(__file__),
                             "../../../tests/proof_of_concept/Graph_Embedding/cached_embeddings/%s_%s.emb" % (
                                     emb,self._id)))):
            with open(os.path.abspath(
                    os.path.join(os.path.dirname(__file__),
                                 "../../../tests/proof_of_concept/Graph_Embedding/cached_embeddings/%s_%s.emb" % (
                                         emb, self._id))), 'rb') as f:
                model = pickle.load(f)
        else:
            model = {}
            for node in G.nodes():
                vector = np.random.rand(emb_dim)
                model[str(node)] = vector
            with open(os.path.abspath(
                    os.path.join(os.path.dirname(__file__),
                                 "../../../tests/proof_of_concept/Graph_Embedding/cached_embeddings/%s_%s.emb" % (
                                         emb,self._id))), 'wb') as f:
                pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
        return model
    def create_embedding(self, G):
        embedding_dict = {'deepwalk': self.deepwalk, 'Random': self.random}
        emb = self.embedding_params.get('name', 'Random')

        return embedding_dict[emb](G)

    # def generate_embedding(self, G):
    #
    #     dim = self.embedding_params.get('dimension', 128)
    #     data = self.embedding_params.get('dataset', 'CiteSeer')
    #     emb = self.embedding_params.get('name', 'Random')
    #
    #     nodes_id = sorted([ int(i) for i in G.nodes()])
    #     if emb == 'Random':
    #         model = {}
    #         for node in nodes_id:
    #             vector = np.random.rand(dim)
    #             model[str(node)] = vector
    #     else:
    #         from gensim.models import KeyedVectors
    #         path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../res/local_files"))
    #         emb_file = path + '/%s_%s.%s' % (data, dim, emb)
    #         model = KeyedVectors.load_word2vec_format(emb_file, binary=False)
    #
    #     return model

    def fit(self,X,y=None):

        from sklearn.preprocessing import StandardScaler
        import sklearn.pipeline as pipeline
        import sklearn.metrics as metrics
        import pandas as pd

        edge_functions = {
            "hadamard": lambda a, b: a * b,
            "average": lambda a, b: 0.5 * (a + b),
            "l1": lambda a, b: np.abs(a - b),
            "l2": lambda a, b: np.abs(a - b) ** 2,
        }

        emb_dim = self.embedding_params.get('dimension', 128)
        emb = self.embedding_params.get('name')
        dataset = self.embedding_params.get('dataset')
        prop_pos = self.graph_params.get('prop_pos', 0.5)
        prop_neg = self.graph_params.get('prop_neg', 0.5)

        Gtrain = self.create_graph(X)
        Gtrain, train_pos_edge_list, train_neg_edge_list = self.generate_pos_neg_links(Gtrain,prop_pos,prop_neg)

        edges_train , labels_train = self.get_selected_edges(train_pos_edge_list,train_neg_edge_list)

        Gtest = self.create_graph(X)
        Gtest, test_pos_edge_list, test_neg_edge_list = self.generate_pos_neg_links(Gtest, prop_pos, prop_neg)

        edges_test, labels_test = self.get_selected_edges(test_pos_edge_list,test_neg_edge_list)

        # G = self.create_graph(X)

        model = self.create_embedding(Gtrain)

        result = open(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../tests/proof_of_concept/Graph_Embedding/results/link_prediction_%s_%s_%s.txt"%(dataset,emb,emb_dim))),'w')
        aucs = {func: [] for func in edge_functions}

        for iter in  range(self.num_iteration):
            print("Iteration %d of %d" % (iter+1, self.num_iteration))

            for edge_fn_name, edge_fn in edge_functions.items():

                edge_features_train = self.edges_to_features(edges_train, edge_fn, emb_dim, model)
                edge_features_test = self.edges_to_features(edges_test, edge_fn, emb_dim,model)

                #Classifier
                scaler = StandardScaler()
                clf = pipeline.make_pipeline(scaler,self.classifier)

                #fit classifier
                clf.fit(edge_features_train, labels_train)

                #test classifier
                auc_test = metrics.scorer.roc_auc_scorer(clf,edge_features_test, labels_test)
                print(edge_fn_name, emb, emb_dim, '%.3f'% auc_test)
                aucs[edge_fn_name].append(auc_test)

        for edge_fn_name in edge_functions.keys():
            result.write(str(dataset) + ' Embedding:  ' + str(emb) + ' Embedding_dimension:  ' + str(
                emb_dim) + ' Edge Function:  ' + str(edge_fn_name) + '    AUC: ' + str('%.3f' % np.mean(aucs[edge_fn_name])))
            result.write('\n')
        result.close()

    def load_components(self):
        """
        The function loads different components required for the wrapper such embeddings, estimators
        :return: Boolean indicating the success of the function
        """
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../res/mapping"))
        with open(path+'/mapping.json') as f:
            framework_dict = json.load(f)
        algorithms = framework_dict.get('algorithms', None)
        self.algorithms_dict = dict()
        for alg in algorithms:
            self.algorithms_dict[alg['name']] = dict()
            alg_dict = dict()
            alg_dict['path'] = alg["implementation"]['scikit-learn']
            alg_params = dict()
            model_parameters = alg['hyper_parameters'].get('model_parameters')
            optimisation_params = alg['hyper_parameters'].get('optimisation_parameters')
            for param in model_parameters:
                # alg_params[param['name']] = dict()
                params_dict = dict()
                params_dict['optional'] = param['optional']
                params_dict['default'] = param['scikit-learn']['default_value']
                alg_params[param['scikit-learn']['path']] = copy.deepcopy(params_dict)
            for param in optimisation_params:
                # alg_params[param['name']] = dict()
                params_dict = dict()
                params_dict['optional'] = param['optional']
                params_dict['default'] = param['scikit-learn']['default_value']
                alg_params[param['scikit-learn']['path']] = copy.deepcopy(params_dict)
            alg_dict['params'] = copy.deepcopy(alg_params)
            self.algorithms_dict[alg['name']] = copy.deepcopy(alg_dict)

        if self.algorithms_dict == dict():
            return False
        return True