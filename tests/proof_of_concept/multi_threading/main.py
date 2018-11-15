
from sklearn.externals.joblib import Parallel, delayed
from sklearn.svm import LinearSVC, SVC, LinearSVR, SVR
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.manifold import Isomap
from sklearn.neighbors import KNeighborsRegressor
import time

EXPERIMENT_ARRAY = []

class Experiment:

    def __init__(self, workflow):
        self._workflow = workflow

    def fit(self, x, y):
        self._workflow.fit(x, y)

def train_model(X, y, seed):
    '''
    model = LinearSVR()
    return model.fit(X, y)
    '''
    workflow = Pipeline([('scl', StandardScaler()),
                         ('pca', PCA(n_components=2)),
                         ('clf', KNeighborsRegressor())
                         ])


    #estimators = [('clf', SVR(C=float(seed/100 + 0.001)))]
    #workflow = Pipeline(pipe_svm)
    ex = Experiment(workflow)
    return ex.fit(X, y)


def train_model_linear(X, y, seed):

    model = LinearSVR(C=float(seed/100 + 0.001))
    return model.fit(X, y)
    '''
    estimators = [('clf', LinearSVR(random_state=seed))]
    workflow = Pipeline(estimators)
    return workflow.fit(X, y)
    '''

seed_max = 100

def run_serial():
    iris = datasets.load_iris()
    prices = datasets.load_boston()
    X = iris.data
    Y = iris.target

    for seed in range(seed_max):
        train_model(X, Y, seed)


def run_parallel():
    iris = datasets.load_iris()
    prices = datasets.load_boston()
    X = iris.data
    Y = iris.target
    result = Parallel(n_jobs=8)(delayed(train_model)(X, Y, seed) for seed in range(seed_max))


c1 = time.time()
run_serial()
c2 = time.time()
print('Execution time for serial:{time_diff}'.format(time_diff=c2-c1))

c1 = time.time()
run_parallel()
c2 = time.time()
print('Execution time for parallel:{time_diff}'.format(time_diff=c2-c1))

#import timeit
#print(timeit.repeat("run_parallel()", "from __main__ import run_parallel", number=10))




