"""
Classes for managing experiments. Contains
- format conversion of data sets (e.g. pandas<-->numpy)
- parameter settings for experiments
- hyperparameter optimisation
- logging
"""
# todo overthink the logger architecture. Maybe the storage should be handled with the exxperiment, and not within
# a particular logger class. so the Experiment could be used to access splits later on and to reproduce
# individual steps.
from padre.base import MetadataEntity, default_logger
from padre.utils import _const
from padre.visitors.scikit.scikitpipeline import SciKitVisitor
import sys
import numpy as np

####################################################################################################################
#  Module Private Functions
####################################################################################################################

def _sklearn_runner():
    pass

def _is_sklearn_pipeline(pipeline):
    """
    checks whether pipeline is a sklearn pipeline
    :param pipeline:
    :return:
    """
    # we do checks via strings, not isinstance in order to avoid a dependency on sklearn
    return type(pipeline).__name__ == 'Pipeline' and type(pipeline).__module__ == 'sklearn.pipeline'




####################################################################################################################
#  API Functions
####################################################################################################################


####################################################################################################################
#  API Classes
####################################################################################################################
class _Phases(_const):

    fitting = "fitting"
    validating = "validating"
    predicting = "predicting"

"""
Enum for the different phases of an experiment
"""
phases = _Phases()


class ExperimentLogger:
    """
    The experiment logger allows to log experimental status information to the server. It can be thought of a structured
    log file.

    """

    def __init__(self, backend=None):
        """
        Logger for experiments
        :param backend: backend for storing the events.
        """
        self._backend = backend

    def _do_print(self):
        return self._backend is None

    def register_experiment(self, experiment):
        if self._do_print():
            print("Registering experiment: " + str(experiment))


    def start_run(self, experiment):
        """
        indicates the start of an experiment run.
        :param experiment:
        :return:
        """
        if self._do_print():
            print("Start experiment")
            print("\t", experiment)

    def stop_run(self, experiment):
        """
        indicates the stop of an experiment run.
        :param experiment:
        :return:
        """
        if self._do_print():
            print("Start experiment")
            print("\t", experiment)

    def start_split(self, experiment, split, train_idx, test_idx, val_idx):
        if self._do_print():
            print("\tStarting split ", split)


    def stop_split(self, experiment, split):
        if self._do_print():
            print("\tStop split ", split)

    def log_event(self, split, phase=None, event=None, is_start=None, **payload):
        if self._do_print():
            print("\t\tEvent", split, phase, event, " (starting=%s)" % (str(is_start)))

    def log_status(self, split, phase, step=None, keys=None, values=None):
        """
        logs training and test events like for example loss functions.
        :param step: int representing the split number
        :param split:
        :param phase: String describing the current phase (should be taken from `phases` variable)
        :param keys: list of keys to log
        :param values: values to the keys. usually every key has a numeric or nominal features
                       (types: int, float, String). Must be the same length as keys
        :return:
        """
        if self._do_print():
            print("\t\tStatus ", split, phase, step, ",".join([str(k)+"="+str(v) for k, v in zip(keys, values)]))

    def log_result(self, split, step, **results):
        if self._do_print():
            print("\t\tResults ", split, step, results)


class SKLearnWorkflow:
    """
    This class encapsulates an sklearn workflow which allows to run sklearn pipelines or a list of sklearn components,
    report the results according to the outcome via the experiment logger.

    A workflow is a single run of fitting, transformation and inference.
    It does not contain any information on the particular split or state of an experiment.
    Workflows are used for abstracting from the underlying machine learning framework.
    """

    def __init__(self, pipeline, step_wise=False):
        # check for final component to determine final results
        # if step wise is true, log intermediate results. Otherwise, log only final results.
        # distingusish between training and fitting in classification.
        self._pipeline = pipeline
        self._step_wise = step_wise

    def fit(self, dataset, train_idx, val_idx, logger, do_scoring=True, split=0):
        # todo split as parameter just for logging is not very good design. Maybe builder pattern would be better?
        if self._step_wise:
            raise NotImplemented()
        else:
            # do logging here
            logger.log_event(split, phase=phases.fitting, is_start=True)
            y = dataset.targets()[train_idx].reshape((len(train_idx),))
            self._pipeline.fit(dataset.features()[train_idx], y)
            logger.log_event(split, phase=phases.fitting, is_start=False)
            if do_scoring and self.is_scorer():
                score = self._pipeline.score(dataset.features()[train_idx], y)
                logger.log_status(split, phases.fitting, keys=["training score"], values=[score])
                if val_idx is not None and len(val_idx) > 0:
                    y = dataset.targets()[val_idx].reshape((len(val_idx),))
                    score = self._pipeline.score(dataset.features()[val_idx], y)
                    logger.log_status(split, phases.fitting, keys=["validation score"], values=[score])

    def infer(self, dataset, test_idx, logger, do_scoring=True, split=0):
        if self._step_wise:
            # step wise means going through every component individually and log their results / timing
            raise NotImplemented()
        else:
            # do logging here
            y = dataset.targets()[test_idx].reshape((len(test_idx),))
            if self.is_inferencer():
                y_predicted = self._pipeline.predict(dataset.features()[test_idx])
                logger.log_result(split, 0, pred=y_predicted, truth=y, confidences=None)
            if do_scoring and self.is_scorer():
                score = self._pipeline.score(dataset.features()[test_idx], y,)
                logger.log_status(split, phases.fitting, keys=["test score"], values=[score])

    def is_inferencer(self):
        return getattr(self._pipeline, "predict", None)

    def is_scorer(self):
        return getattr(self._pipeline, "score", None)

    def is_transformer(self):
        return getattr(self._pipeline, "transform", None)

    def configuration(self):
        return SciKitVisitor(self._pipeline)


class Splitter:
    """
    The splitter creates index arrays into the dataset for different splitting startegies. It provides an iterator
    over the different splits.

    Currently the following splitting strategies are supported:
     - random split (stratified / non-stratified)
     - cross validation (stratified / non-stratified)
    """
    def __init__(self, ds, strategy="random", test_ratio=0.25,
                 val_ratio=0, n_folds=None,
                 stratified=None, random_seed=None,
                 no_shuffle=False):
        self._dataset = ds
        self._num_examples = ds.size[0]
        default_logger.error(strategy == "random" or strategy == "cv", self,
                             f"Unknown splitting strategy {strategy}. Only 'cv' or 'random' allowed")
        self._strategy = strategy
        default_logger.warn(test_ratio is None or (0.0 <= test_ratio <= 1.0), self,
                            f"Wrong ratio of test set provided {test_ratio}. Continuing with default=0")
        self._test_ratio = test_ratio
        default_logger.warn(val_ratio is None or (0.0 <= val_ratio <= 1.0), self,
                            f"Wrong ratio of evaluation set provided {val_ratio}. Continuing with default=0")

        self._val_ratio = val_ratio
        if n_folds is None:
            if strategy == "random":
                self._n_folds = 1
            else:
                self._n_folds = 3
        else:
            self._n_folds = n_folds
        default_logger.error(1 <= self._n_folds, self, f"Number of folds not positive {self._n_folds}")
        self._random_seed = random_seed
        self._no_shuffle = no_shuffle
        default_logger.warn(not (self._n_folds == 1 and self._strategy == "random" and self._no_shuffle), self,
                            f"Random test split will be always the same since shuffling is not permitted")
        default_logger.error(self._n_folds < self._dataset.size[0] or self._strategy != "cv", self,
                             f"There are more folds than examples: {n_folds}<{self._dataset.size[0]}")
        if stratified is None:
            self._stratified = ds.targets() is not None
        else:
            self._stratified = stratified
            if stratified and ds.targets() is None:
                default_logger.warn(False, self,
                                    f"Targets not provided in dataset {ds}. Can not do stratified splitting")
                self._stratified = False

    def splits(self):
        """
        returns an generator function over all available splits. Every iterator returns a triple (train_idx, test_idx, eval_idx)
        where *_idx is an index array for getting the particular split of the training data set (e.g. dataset.data()[train_idx]
        provides the training slice part)
        :return: generator function
        """
        # first create index array and random state vector
        n = self._dataset.size[0]
        r = np.random.RandomState(self._random_seed)
        idx = np.arange(n)

        def splitting_iterator():
            # now apply splitting strategy
            # todo s: time aware cross validation, stratified splits,
            # Todo do sanity checks that indizes do not overlap
            if self._strategy == "random":
                for i in range(self._n_folds):
                    if not self._no_shuffle:  # Reshuffle every "fold"
                        r.shuffle(idx)
                    n_tr = int(n * (1.0-self._test_ratio))
                    train, test = idx[:n_tr], idx[n_tr:]
                    if self._val_ratio > 0:  # create a validation set out of the test set
                        n_v = int(len(train) * self._val_ratio)
                        yield train[:n_v], test, train[n_v:]
                    else:
                        yield train, test, None
            elif self._strategy == "cv":
                for i in range(self._n_folds):
                    n_te = i*int(n / self._n_folds)
                    if i == self._n_folds - 1:
                        upper = []
                        test = range(i*n_te, n)
                    else:
                        upper = list(range(-n+(i+1)*n_te, 0, 1))
                        test = range(i * n_te, (i + 1) * n_te)
                    train, test = idx[list(range(i*n_te))+upper], idx[test]
                    if self._val_ratio > 0:  # create a validation set out of the test set
                        n_v = int(len(train) * self._val_ratio)
                        yield train[:n_v], test, train[n_v:]
                    else:
                        yield train, test, None
            else:
                raise ValueError(f"Unknown splitting strategy {self._splitting_strategy}")

        return splitting_iterator()


class Experiment(MetadataEntity):
    """
    Experiment class covering functionality for executing and evaluating machine learning experiments.

    An experiment requires:
    - a pipeline. A pipeline follows the convention of a sklearn pipeline
    <http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html>,
    i.e. a list of (name, class) tuples, where the class implements:
       - a `fit` function (parameters need to be defined)
       - a `predict` function in case of supervised prediction (parameters need to be defined)
       - a `transform`function in case of feature space transformers (parameters need to be defined)

    - a dataset. First, we only support a single dataset but components might need additional datasets
    - metadata describing
      - the splitting strategy (see http://scikit-learn.org/stable/model_selection.html)
        - uses checked_cv for crossvalidation (and passes all the options to the function)
        - uses train_test_split for random splitting (and passes all the options to the function)

    The parameters and setup of the Experiment are automatically extracted using the visitor functionality.

    All metadata provided to the experiment will be stored along the experiment description. However, the following
    properties will gain special purpose for an experiment:
    - task - determines the task achieved by a experiment (e.g. classification, regression, metric learning etc.)
    - name - determines the name of an experiment
    - description - determines the description of an experiment
    - domain - determines the application domain
    """
    def __init__(self, dataset,
                 workflow,
                 splitter,
                 **options):
        super().__init__(**options)
        self._dataset = dataset
        # todo workflow semantic not clear. Fit and infer is fine, but we need someting for transform
        if _is_sklearn_pipeline(workflow):
            self._pipeline = SKLearnWorkflow(workflow, options.pop("sk_learn_stepwise", False))
        else:
            self._pipeline = workflow

        self._splitting = splitter
        self._options = options
        self._fill_sys_info()

    def _fill_sys_info(self):
        # Todo implement gaterhing of system info and stroing it as metadata
        self._metadata["sys_info"] = "Not implemented yet"


    def configuration(self):
        return self._pipeline.configuration()

    def hyperparameters(self):
        """
        returns the hyperparameters per pipeline element as dict from the extracted configruation
        :return:
        """
        params = dict()
        steps = self.configuration()[0]["steps"]
        for step in steps:
            params = dict(step)
            if "doc" in params:
                del params["doc"]
        return params


    @staticmethod
    def create(config):
        """
         create an experiment from a provided configuration dictionary
        :param config: dictionary containing the configuration
        :return:
        """
        pass

    def run(self, logger=None):
        """
        runs the experiment
        :param reporter:
        :return:
        """
        # todo allow split wise execution of the individual workflow steps. some kind of reproduction / debugging mode
        # which gives access to one split, the model of the split etc.

        if logger is None:
            logger = ExperimentLogger()
        # register experiment through logger
        logger.register_experiment(self)
        logger.start_run(self)
        # log run start here.
        for split, (train_idx, test_idx, val_idx) in enumerate(self._splitting.splits()):
            self._pipeline.fit(self._dataset, train_idx, val_idx, logger, True, split)
            if self._pipeline.is_inferencer() and test_idx is not None and len(test_idx)>0:
                self._pipeline.infer(self._dataset, test_idx, logger, True, split)

    def access(self):
        """
        placeholder for access functionality. to be defined later
        :return:
        """
        pass
