"""
Classes for managing experiments.

An experiment is represented by the `Experiment` class and the main entry point for using the module. A
The experiment class creates Runs, which are instantiations of the workflow over a dataset.
Every run conducts splits of the dataset, which maybe a trivial split (e.g. no splitting) and aggregates scores from
every split.

A core design principle is based on having one configuration, the does the setup and control of the experiment.
The worklfow itself is provide from the outside and needs to implement  a `fit` method and a `infer` method.
Future implementations might implement a joint `execution` method. The method is provided with a split object,
that contains the execution environment including logging facilities. so it can be seen as context.

Ideally, the following piece of code realises a workflow

.. code-block::python

   ex = Experiment(my_config_as_dict)

   @ex.fit
   def my_super_ml_trainer(_context):
      train(_context.train)
      _context.log(xy)

   ex.run()


todo: we can put a user specific context in the `my_config_dict` which can be then accessed through `_context`
"""
import itertools
import platform
# todo overthink the logger architecture. Maybe the storage should be handled with the exxperiment, and not within
# a particular logger class. so the Experiment could be used to access splits later on and to reproduce
# individual steps.
from time import time

import numpy as np

import padre.visitors.parameter
from padre.base import MetadataEntity, default_logger, result_logger
from padre.utils import _const
from padre.visitors.scikit import SciKitVisitor


####################################################################################################################
#  Module Private Functions and Classes
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


class _timer_priorities(_const):
    """
    Constant class detailing the different priorites possible for a timer.
    """
    NO_LOGGING = 0
    HIGH_PRIORITY = 1
    MED_PRIORITY = 2
    LOW_PRIORITY = 3


class _timer_defaults(_const):
    """
    Constant class detailing the different default values for a timer.
    """
    DEFAULT_PRIORITY = 3
    DEFAULT_TIMER_DESCRIPTION = None
    DEFAULT_TIMER_NAME = 'default_timer'
    DEFAULT_TIME = 0.0


timer_priorities = _timer_priorities
timer_defaults = _timer_defaults


class TimeKeeper:
    """
    This class creates a dictionary of timers.
    It has a log timer function that would log each timer passed to it.
    If the timer name is already present in the dictionary, the timer is
    popped out and the execution time is calculated. Multiple timers can be kept
    track of this way.
    Priorities of the timers are also defined
    The timer is logged only if the priority of the timer is equal to or higher than the
    priority defined while initializing.
    Priorities Available.
    NO_LOGGING: None of the timers are logged.
    HIGH_PRIORITY: Only timers having high priority are logged.
    MED_PRIORITY: Only timers with medium or higher priority are logged.
    LOW_PRIORITY: All timers are logged.
    """

    def __init__(self, priority):
        """
        This initializes the TimeKeeper class.
        :param priority: The current logging priority
        """
        self._timers = dict()
        self._priority = priority

    def __del__(self):
        """
        This is the destructor function of the TimeKeeper class.
        The purpose of the destructor is to check whether any timers are left in the TimeKeeper class at
        the end of execution.
        :return: None
        """
        if len(self._timers) > 0:
            print("Error: The following Timers still present in the list")
            for key in self._timers:
                print(key)

    def log_timer(self, timer_name=timer_defaults.DEFAULT_TIMER_NAME,
                  priority=timer_defaults.DEFAULT_PRIORITY,
                  description=timer_defaults.DEFAULT_TIMER_DESCRIPTION):
        """

        :param timer_name: Name of the unique timer. If timer is already present,
        the timer would be popped out and duration recorded.
        :param priority: priority of the timer.
        :param description: The string description of what the timer measures.
        :return: Description of the the timer and its duration.
        """
        # If there is no timer by the name, add the timer to the dictionary
        # Else pop out the timer, and check whether the priority of the timer is
        # equal to or higher than the priority of the program.
        # If it is, then print the description and timer
        if self._timers.get(timer_name) is None:
            new_timer = TimerContents(priority=priority,
                                      description=description,
                                      curr_time=time())
            self._timers[timer_name] = new_timer

        else:
            old_timer = self._timers.pop(timer_name, None)
            if old_timer.get_timer_priority() <= self._priority:
                return old_timer.get_description(), time() - old_timer.get_time()

    def start_timer(self, timer_name=timer_defaults.DEFAULT_TIMER_NAME,
                    priority=timer_defaults.DEFAULT_PRIORITY,
                    description=timer_defaults.DEFAULT_TIMER_DESCRIPTION):
        """
        Starts a unique timer with the key as timer_name
        :param timer_name: Unique name of the timer
        :param priority: Priority of the timer
        :param description: Describes the purpose of the timer
        :return: None
        """
        if self._timers.get(timer_name) is None:
            new_timer = TimerContents(priority=priority,
                                      description=description,
                                      curr_time=time())
            self._timers[timer_name] = new_timer

    def stop_timer(self, timer_name):
        """
        Stops a timer and measures its duration
        :param timer_name:
        :return: Description of the timer and its duration
                 None if a timer with its unique name is not present
        """
        if timer_name is None:
            return None

        old_timer = self._timers.pop(timer_name, None)
        if old_timer.priority <= self._priority:
            return old_timer.description, time() - old_timer.time


class TimerContents:
    """
    This class contains the contents to be displayed and calculated,
    using the TimeKeeper Class.
    The class stores three values and there are three get attributes
    corresponding to each value. There is no set attribute and the values are
    initialized during object creation itself.
    """

    def __init__(self, priority=timer_defaults.DEFAULT_PRIORITY,
                 description=timer_defaults.DEFAULT_TIMER_DESCRIPTION,
                 curr_time=timer_defaults.DEFAULT_TIME):
        """
        The initialization of the Timer class. All the arguments are given default values
        which are present in timer_defaults.
        :param priority: The priority of the timer.
        :param description: The description of the timer.
        :param curr_time: The time to start logging.
        """
        self._timer_desc = description
        self._timer_priority = priority
        self._time = curr_time

    @property
    def description(self):
        return self._timer_desc

    @property
    def time(self):
        return self._time

    @property
    def priority(self):
        return self._timer_priority


# TODO: A better way of using the default timer
# A static object shared throughout the instances of _LoggerMixin
default_timer = TimeKeeper(timer_defaults.DEFAULT_PRIORITY)


class _LoggerMixin:
    """
    Mixin that provides function for logging and storing events in the backend.
    """

    _backend = None
    _stdout = False
    _events = {}
    _overwrite = True

    def _padding(self, source):
        if isinstance(source, Split):
            return "\t\t"
        elif isinstance(source, Run):
            return "\t"
        else:
            return ""

    def log_start_experiment(self, experiment):
        if self.has_backend():
            # todo overwrite solution not good. we need a mechanism to add runs
            self._backend.put_experiment(experiment, allow_overwrite=self._overwrite)
        self.log_event(experiment, exp_events.start, phase=phases.experiment)

    def log_stop_experiment(self, experiment):
        self.log_event(experiment, exp_events.stop, phase=phases.experiment)

    def log_start_run(self, run):
        if self.has_backend():
            self._backend.put_run(run.experiment, run)
        self.log_event(run, exp_events.start, phase=phases.run)

    def log_stop_run(self, run):
        self.log_event(run, exp_events.stop, phase=phases.run)

    def log_start_split(self, split):
        if self.has_backend():
            self._backend.put_split(split.run.experiment, split.run, split)
        self.log_event(split, exp_events.start, phase=phases.split)

    def log_stop_split(self, split):
        self.log_event(split, exp_events.stop, phase=phases.split)

    def log_event(self, source, kind=None, **parameters):
        # todo signature not yet fixed. might change. unclear as of now
        if kind == exp_events.start and source is not None:
            # self._events[source] = time()
            # Create a unique id for the timer.
            # Currently creating it by self._id + phase parameter
            # TODO: A better way for creating identifiers for each phase
            # TODO: Pass description of time too if needed
            timer_name = str(self._id)
            timer_description = ''
            phase = parameters.get('phase', None)
            if phase is not None:
                timer_name = timer_name + str(phase)

            timer_description = parameters.get('description', None)
            default_timer.start_timer(timer_name, timer_priorities.HIGH_PRIORITY, timer_description)
        elif kind == exp_events.stop and source is not None:
            # if source in self._events:
            # parameters["duration"] = time() - self._events[source]
            # Creation of unique identifier to get back the time duration
            timer_name = str(self._id)
            phase = parameters.get('phase', None)
            if phase is not None:
                timer_name = timer_name + str(phase)
            description, duration = default_timer.stop_timer(timer_name)
            if description is not None:
                parameters['description'] = description
            parameters['duration'] = duration

        if self._stdout:
            default_logger.log(source, "%s: %s" % (str(kind),
                                                   "\t".join([str(k) + "=" + str(v) for k, v in parameters.items()])),
                               self._padding(source))

    def log_score(self, source, **parameters):
        # todo signature not yet fixed. might change. unclear as of now
        if self._stdout:
            default_logger.log(source, "%s" % ("\t".join([str(k) + "=" + str(v) for k, v in parameters.items()]))
                               , self._padding(source))

    def log_stats(self, source, **parameters):
        # todo signature not yet fixed. might change. unclear as of now
        if self._stdout:
            default_logger.log(source, "%s" % ("\t".join([str(k) + "=" + str(v) for k, v in parameters.items()])),
                               self._padding(source))

    def log_result(self, source, **parameters):
        # todo signature not yet fixed. might change. unclear as of now
        if self._stdout:
            default_logger.log(source, "%s" % ("\t".join([str(k) + "=" + str(v) for k, v in parameters.items()])),
                               self._padding(source))

    def has_backend(self):
        return self._backend is not None

    @property
    def backend(self):
        return self._backend

    @property
    def stdout(self):
        return self._stdout

    @backend.setter
    def backend(self, backend):
        self._backend = backend


####################################################################################################################
#  API Functions
####################################################################################################################


####################################################################################################################
#  API Classes
####################################################################################################################
class _Phases(_const):
    experiment = "experiment"
    run = "run"
    split = "split"
    fitting = "fitting/training"
    validating = "validating"
    inferencing = "inferencing/testing"


"""
Enum for the different phases of an experiment
"""
phases = _Phases()


class _ExperimentEvents(_const):
    start = "start"
    stop = "stop"


"""
Enum for the different phases of an experiment
"""
exp_events = _ExperimentEvents()


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

    def fit(self, ctx):
        # todo split as parameter just for logging is not very good design. Maybe builder pattern would be better?
        if self._step_wise:
            raise NotImplemented()
        else:
            # do logging here
            ctx.log_event(ctx, kind=exp_events.start, phase="sklearn." + phases.fitting)
            y = ctx.train_targets.reshape((len(ctx.train_targets),))
            self._pipeline.fit(ctx.train_features, y)
            ctx.log_event(ctx, kind=exp_events.stop, phase="sklearn." + phases.fitting)
            if self.is_scorer():
                ctx.log_event(ctx, kind=exp_events.start, phase="sklearn.scoring.trainset")
                score = self._pipeline.score(ctx.train_features, y)
                ctx.log_event(ctx, kind=exp_events.stop, phase="sklearn.scoring.trainset")
                ctx.log_score(ctx, keys=["training score"], values=[score])
                if ctx.has_valset():
                    y = ctx.val_targets.reshape((len(ctx.val_targets),))
                    ctx.log_event(ctx, kind=exp_events.start, phase="sklearn.scoring.valset")
                    score = self._pipeline.score(ctx.val_features, y)
                    ctx.log_event(ctx, kind=exp_events.stop, phase="sklearn.scoring.valset")
                    ctx.log_score(ctx, keys=["validation score"], values=[score])

    def infer(self, ctx, train_idx, test_idx):
        if self._step_wise:
            # step wise means going through every component individually and log their results / timing
            raise NotImplemented()
        else:
            # do logging here
            if ctx.has_testset() and self.is_inferencer():
                y = ctx.test_targets.reshape((len(ctx.test_targets),))
                # todo: check if we can estimate probabilities, scores or hard decisions
                # this also changes the result type to be written.
                # if possible, we will always write the "best" result type, i.e. which retains most information (
                # if
                y_predicted = self._pipeline.predict(ctx.test_features)
                results = {'predicted': y_predicted.tolist(),
                           'truth': y.tolist()}

                ctx.log_result(ctx, mode="probability", pred=y_predicted, truth=y,
                               probabilities=None, scores=None,
                               transforms=None, clustering=None)
                # log the probabilities of the result too if the method is present
                if 'predict_proba' in dir(self._pipeline.steps[-1][1]):
                    y_predicted_probabilities = self._pipeline.predict_proba(ctx.test_features)
                    ctx.log_result(ctx, mode="probabilities", pred=y_predicted,
                                   truth=y, probabilities=y_predicted_probabilities,
                                   scores=None, transforms=None, clustering=None)
                    results['probabilities'] = y_predicted_probabilities.tolist()
                    metrics = dict()
                    # Calculate the confusion matrix
                    confusion_matrix = self.compute_confusion_matrix(Predicted=y_predicted.tolist(),
                                                                     Truth=y.tolist())
                    metrics['confusion_matrix'] = confusion_matrix
                    result_logger.log_metrics(metrics=metrics)

                if self.is_scorer():
                    score = self._pipeline.score(ctx.test_features, y, )
                    ctx.log_score(ctx, keys=["test score"], values=[score])

                results['train_idx'] = train_idx
                results['test_idx'] = test_idx

                result_logger.log_result(results)

    def is_inferencer(self):
        return getattr(self._pipeline, "predict", None)

    def is_scorer(self):
        return getattr(self._pipeline, "score", None)

    def is_transformer(self):
        return getattr(self._pipeline, "transform", None)

    def configuration(self):
        return SciKitVisitor(self._pipeline)

    def compute_confusion_matrix(self, Predicted=None,
                                 Truth=None):
        """
        This function computes the confusionmatrix of a classification result.
        This was done as a general purpose implementation of the confusion_matrix
        :param Predicted: The predicted values of the confusion matrix
        :param Truth: The truth values of the confusion matrix
        :return: The confusion matrix
        """
        import copy
        if Predicted is None or Truth is None or \
                len(Predicted) != len(Truth):
            return None

        # Get the number of labels from the predicted and truth set
        label_count = len(set(Predicted).union(set(Truth)))
        confusion_matrix = np.zeros(shape=(label_count, label_count), dtype=int)
        # If the labels given do not start from 0 and go up to the label_count - 1,
        # a mapping function has to be created to map the label to the corresponding indices
        if (min(Predicted) != 0 and min(Truth) != 0) or \
                (max(Truth) != label_count - 1 and max(Predicted) != label_count - 1):
            labels = list(set(Predicted).union(set(Truth)))
            for idx in range(0, len(Truth)):
                row_idx = int(labels.index(Predicted[idx]))
                col_idx = int(labels.index(Truth[idx]))
                confusion_matrix[row_idx][col_idx] += 1

        else:

            # Iterate through the array and update the confusion matrix
            for idx in range(0, len(Truth)):
                confusion_matrix[int(Predicted[idx])][int(Truth[idx])] += 1

        return copy.deepcopy(confusion_matrix.tolist())


class Splitter:
    """
    The splitter creates index arrays into the dataset for different splitting startegies. It provides an iterator
    over the different splits.

    Currently the following splitting strategies are supported:
     - random split (stratified / non-stratified). If no_shuffle is true, the order will not be changed.
     - cross validation (stratified / non-stratified)"
     - explicit - expects an explicit split given as parameter indices = (train_idx, val_idx, test_idx)
     - function - expects a function taking as input the dataset and performing the split.
     - none - there will be no splitting. only a training set will be provided

     Options:
     ========
     - strategy={"random"|"cv"|"explicit"|"none"/None} splitting strategy, default random
     - test_ratio=float[0:1]   ratio of test dataset, default 0.25
     - val_ratio=float[0:1]    ratio of the validation test (taken from the training set), default 0
     - n_folds=int             number of folds when selecting cv strategies, default 3. smaller than dataset size
     - random_seed=int         seed for the random generator or None if no seeding should be done
     - stratified={True|False|None} True, if splits should consider class stratification. If None, than stratification
                               is activated when there are targets (default). Otherwise, stratification strategies
                                is taking explicityly into account
     - no_shuffle={True|False} indicates, whether shuffling the data is allowed.
     - indices = [(train, validation, test)] a list of tuples with three index arrays in the dataset.
                               Every index array contains
                               the row index of the datapoints contained in the split
     - fn                      function of the form fn(dataset, **options) that returns a iterator over
                               (train, validation, test) tuples (the form is similar to the indices parameter) as split
    """

    def __init__(self, ds, **options):
        self._dataset = ds
        self._num_examples = ds.size[0]
        self._strategy = options.pop("st"
                                     ""
                                     ""
                                     "rategy", "random")
        default_logger.error(self._strategy == "random" or self._strategy == "cv", self,
                             f"Unknown splitting strategy {self._strategy}. Only 'cv' or 'random' allowed")
        self._test_ratio = options.pop("test_ratio", 0.25)
        default_logger.warn(self._test_ratio is None or (0.0 <= self._test_ratio <= 1.0), self,
                            f"Wrong ratio of test set provided {self._test_ratio}. Continuing with default=0")
        self._val_ratio = options.pop("val_ratio", 0)
        default_logger.warn(self._val_ratio is None or (0.0 <= self._val_ratio <= 1.0), self,
                            f"Wrong ratio of evaluation set provided {self._val_ratio}. Continuing with default=0")

        self._n_folds = options.pop("n_folds", 3)
        default_logger.error(1 <= self._n_folds, self, f"Number of folds not positive {self._n_folds}")
        self._random_seed = options.pop("random_seed", None)
        self._no_shuffle = options.pop("no_shuffle", False)
        default_logger.warn(not (self._n_folds == 1 and self._strategy == "random" and self._no_shuffle), self,
                            f"Random test split will be always the same since shuffling is not permitted")
        default_logger.error(self._n_folds < self._dataset.size[0] or self._strategy != "cv", self,
                             f"There are more folds than examples: {self._n_folds}<{self._dataset.size[0]}")
        self._stratified = options.pop("stratified", None)
        self._indices = options.pop("indices", None)
        if self._strategy == "indices":
            default_logger.error(self._indices is not None, self,
                                 f"Splitting strategy {self._strategy} requires an "
                                 f"explicit split given by parameter 'indices'")
        if self._stratified is None:
            self._stratified = ds.targets() is not None
        else:
            if self._stratified and ds.targets() is None:
                default_logger.warn(False, self,
                                    f"Targets not provided in dataset {ds}. Can not do stratified splitting")
                self._stratified = False
        self._splitting_fn = options.pop("fn", None)
        if self._strategy == "function":
            default_logger.error(self._splitting_fn is not None, self,
                                 f"Splitting strategy {self._strategy} requires a function provided via paraneter 'fn'")

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
            if self._strategy is None:
                yield idx, None, None
            elif self._strategy == "explicit":
                for i in self._indices:
                    yield i
            elif self._strategy == "function":
                return self._splitting_fn
            elif self._strategy == "random":
                # for i in range(self._n_folds):
                if not self._no_shuffle:  # Reshuffle every "fold"
                    r.shuffle(idx)
                n_tr = int(n * (1.0 - self._test_ratio))
                train, test = idx[:n_tr], idx[n_tr:]
                if self._val_ratio > 0:  # create a validation set out of the test set
                    n_v = int(len(train) * self._val_ratio)
                    yield train[:n_v], test, train[n_v:]
                else:
                    yield train, test, None
            elif self._strategy == "cv":
                for i in range(self._n_folds):
                    n_te = i * int(n / self._n_folds)
                    if i == self._n_folds - 1:
                        upper = []
                        test = range(i * n_te, n)
                    else:
                        upper = list(range(-n + (i + 1) * n_te, 0, 1))
                        test = range(i * n_te, (i + 1) * n_te)
                    train, test = idx[list(range(i * n_te)) + upper], idx[test]
                    if self._val_ratio > 0:  # create a validation set out of the test set
                        n_v = int(len(train) * self._val_ratio)
                        yield train[:n_v], test, train[n_v:]
                    else:
                        yield train, test, None
            else:
                raise ValueError(f"Unknown splitting strategy {self._splitting_strategy}")

        return splitting_iterator()


class Split(MetadataEntity, _LoggerMixin):
    """
    A split is a single part of a run and the actual excution over parts of the dataset.
    According to the experiment setup the pipeline/workflow will be executed
    """

    def __init__(self, run, num, train_idx, val_idx, test_idx, **options):
        self._run = run
        self._num = num
        self._backend = run.backend
        self._stdout = run.stdout
        self._train_idx = train_idx
        self._val_idx = val_idx
        self._test_idx = test_idx
        self._keep_splits = options.pop("keep_splits", False)
        self._splits = []
        self._id = options.pop("split_id", None)
        super().__init__(self._id, **options)

    @property
    def number(self):
        return self._num

    @property
    def run(self):
        return self._run

    def execute(self):
        self.log_start_split(self)
        # log run start here.
        workflow = self._run.experiment.workflow
        self.log_event(self, exp_events.start, phase=phases.fitting)
        workflow.fit(self)
        self.log_event(self, exp_events.stop, phase=phases.fitting)
        if workflow.is_inferencer() and self.has_testset():
            self.log_event(self, exp_events.start, phase=phases.inferencing)
            workflow.infer(self, self.train_idx.tolist(), self.test_idx.tolist())
            self.log_event(self, exp_events.stop, phase=phases.inferencing)
        self.log_stop_split(self)

    def has_testset(self):
        return self._test_idx is not None and len(self._test_idx) > 0

    def has_valset(self):
        return self._val_idx is not None and len(self._val_idx) > 0

    def has_targets(self):
        return self.dataset.targets() is not None

    @property
    def train_idx(self):
        return self._train_idx

    @property
    def test_idx(self):
        return self._test_idx

    @property
    def val_idx(self):
        return self._val_idx

    @property
    def dataset(self):
        return self._run.experiment.dataset

    @property
    def train_features(self):
        return self.dataset.features()[self._train_idx]

    @property
    def test_features(self):
        if not self.has_testset():
            return None
        else:
            return self.dataset.features()[self._test_idx]

    @property
    def val_features(self):
        if not self.has_valset():
            return None
        else:
            return self.dataset.features()[self._val_idx]

    @property
    def train_targets(self):
        if not self.has_targets():
            return None
        else:
            return self.dataset.targets()[self._train_idx]

    @property
    def test_targets(self):
        if not self.has_testset() or not self.has_targets():
            return None
        else:
            return self.dataset.targets()[self._test_idx]

    @property
    def val_targets(self):
        if not self.has_valset() or not self.has_targets():
            return None
        else:
            return self.dataset.targets()[self._val_idx]

    @property
    def train_data(self):
        return self.dataset.data()[self._train_idx]

    @property
    def test_data(self):
        if not self.has_testset():
            return None
        else:
            return self.dataset.data()[self._test_idx]

    @property
    def val_data(self):
        if not self.has_valset():
            return None
        else:
            return self.dataset.data()[self._val_idx]

    def __str__(self):
        s = []
        if self.id is not None:
            s.append("id:" + str(self.id))
        if self.name is not None and self.name != self.id:
            s.append("name:" + str(self.name))
        if len(s) == 0:
            return str(super())
        else:
            return "Split<" + ";".join(s) + ">"


class Run(MetadataEntity, _LoggerMixin):
    """
    A run is a single instantiation of an experiment with a definitive set of parameters.
    According to the experiment setup the pipeline/workflow will be executed
    """

    def __init__(self, experiment, workflow, **options):
        self._experiment = experiment
        self._workflow = workflow
        self._backend = experiment.backend
        self._stdout = experiment.stdout
        self._keep_splits = options.pop("keep_splits", False)
        self._splits = []
        self._id = options.pop("run_id", None)
        super().__init__(self._id, **options)

    def do_splits(self):
        self.log_start_run(self)
        # instantiate the splitter here based on the splitting configuration in options
        splitting = Splitter(self._experiment.dataset, **self._metadata)
        for split, (train_idx, test_idx, val_idx) in enumerate(splitting.splits()):
            sp = Split(self, split, train_idx, val_idx, test_idx, **self._metadata)
            sp.execute()
            if self._keep_splits:
                self._splits.append(sp)
        self.log_stop_run(self)

    @property
    def experiment(self):
        return self._experiment

    def __str__(self):
        s = []
        if self.id is not None:
            s.append("id:" + str(self.id))
        if self.name is not None and self.name != self.id:
            s.append("name:" + str(self.name))
        if len(s) == 0:
            return str(super())
        else:
            return "Run<" + ";".join(s) + ">"


class Experiment(MetadataEntity, _LoggerMixin):
    """
    Experiment class covering functionality for executing and evaluating machine learning experiments.
    It is determined by a pipeline which is evaluated over a dataset with several configuration.
    A run applies one configuration over the data, which can be splitted in several sub-runs on different dataset parts
    in order to get reliable statistical estimates.

    An experiment requires:
    1. a pipeline / workflow. A workflow implements `fit`, `infer` and `transform` methods, comparable to sklearn.
    Currently, we only support sklearn pipelines, which are wrapped by the SKLearnWorkflow
    <http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html>,
    i.e. a list of (name, class) tuples, where the class implements:
       - a `fit` function (parameters need to be defined)
       - a `infer` function in case of supervised prediction (parameters need to be defined)
       - a `transform`function in case of feature space transformers (parameters need to be defined)

    2. a dataset. An experiment is always tight to one dataset which is the main dataset for running the experiment.
       Future work should allow to include auxiliary resources, but currently we only support one dataset.
    3. metadata describing different aspects of the workflow.
      - the splitting strategy (see Splitter)
      - hyperparameter ranges
      - output control etc.

    4. a backend, which provides functionality for logging experiment progress.


    Experiment Metadata:
    ====================

    All metadata provided to the experiment will be stored along the experiment description. However, the following
    properties will gain special purpose for an experiment:
    - task - determines the task achieved by a experiment (e.g. classification, regression, metric learning etc.)
    - name - determines the name of an experiment
    - id - determines the repository id of an experiment (might be equal to the name, if the name is also the id)
    - description - determines the description of an experiment
    - domain - determines the application domain

    Parameters required:
    ===================
    The following parameters need to be set in the constructor or via annotations
    - dataset : padre.datasets.Dataset
    - workflow: either a padre.experiment.Workflow object or a SKLearn Pipeline
    - backend : backend where the logs, experiments etc. shoudl be written too

    Options supported:
    ==================
    - stdout={True|False} logs event messages to default_logger. Default = True
    - keep_splits={True|False} if true, all split data for every run is kept (including the model, split inidices and training data)
                               are kept in memory. If false, no split data is kept
    - keep_runs={True|False} if true, all rund data (i.e. scores) will be kept in memory. If false, no split run data is not kept
    - n_runs = int  number of runs to conduct. todo: needs to be extended with hyperparameter search

    TODO:
    - Queuing mode
    - Searching Hyperparameter Space

    """

    def __init__(self,
                 **options):
        self._dataset = options.pop("dataset", None)
        # we need to store the dataset_id in the metadata. otherwise, this information might be lost during storage
        options["dataset_id"] = self._dataset.id
        # todo workflow semantic not clear. Fit and infer is fine, but we need someting for transform
        workflow = options.pop("workflow", None)
        self._backend = options.pop("backend", None)
        self._stdout = options.get("stdout", True)
        self._keep_runs = options.get("keep_runs", False) or options.get("keep_splits", False)
        self._runs = []
        self._sk_learn_stepwise = options.get("sk_learn_stepwise", False)
        self._set_workflow(workflow)
        self._last_run = None
        super().__init__(options.pop("ex_id", None), **options)

        self._fill_sys_info()

    def _fill_sys_info(self):
        # TODO: Implement the gathering of system information as dynamic code
        # TODO: Remove hard coded strings.
        # This function collects all system related info in a dictionary
        sys_info = dict()
        sys_info["processor"] = platform.processor()
        sys_info["machine"] = platform.machine()
        sys_info["system"] = platform.system()
        sys_info["platform"] = platform.platform()
        sys_info["platform_version"] = platform.version()
        sys_info["node_name"] = platform.node()
        sys_info["python_version"] = platform.python_version()
        self._metadata["sys_info"] = sys_info

    def _set_workflow(self, w):
        if _is_sklearn_pipeline(w):
            self._workflow = SKLearnWorkflow(w, self._sk_learn_stepwise)
        else:
            self._workflow = w

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, ds):
        self._dataset = ds

    @property
    def backend(self):
        return self._backend

    @backend.setter
    def backend(self, b):
        self._backend = b

    @property
    def workflow(self):
        return self._workflow

    @workflow.setter
    def workflow(self, w):
        self._set_workflow(w)

    def configuration(self):
        return self._workflow.configuration()

    def hyperparameters(self):
        """
        returns the hyperparameters per pipeline element as dict from the extracted configruation
        :return:
        """
        # todo only list the hyperparameters and, if available, the potential value ranges
        # todo experiment.hyperparameters() should deliver a json serialisable object.
        # make it as close to the http implementation as possible
        params = dict()
        steps = self.configuration()[0]["steps"]
        # Params is a dictionary of hyper parameters where the key is the zero-indexed step number
        # The traverse_dict function traverses the dictionary in a recursive fashion and replaces
        # any instance of <class 'padre.visitors.parameter.Parameter'> type to a sub-dictionary of
        # value and attribute. This allows the dictionary to be JSON serializable
        for idx, step in enumerate(steps):
            params["".join(["Step_", str(idx)])] = self.traverse_dict(dict(step))
        return params

    def set_hyperparameters(self, hyperparameters):
        # todo placeholder as loading an experiment should include loading hyperparameters.
        # Howver, in sklearn, the hyperparameters are defined via the pipeline. As long as
        # we do not integrate a second framework, we do not need the mechanism
        pass

    @property
    def workflow(self):
        return self._workflow

    @property
    def dataset(self):
        return self._dataset

    def run(self, append_runs: bool = False):
        """
        runs the experiment
        :param append_runs: If true, the runs will be appended if the experiment exists already.
        Otherwise, the experiment will be deleted
        :return:
        """
        # todo allow split wise execution of the individual workflow steps. some kind of reproduction / debugging mode
        # which gives access to one split, the model of the split etc.
        # todo allow to append runs for experiments
        # register experiment through logger
        self.log_start_experiment(self)
        # todo here we do the hyperparameter search, e.g. GridSearch. so there would be a loop over runs here.
        r = Run(self, self._workflow, **dict(self._metadata))
        r.do_splits()
        if self._keep_runs:
            self._runs.append(r)
        self._last_run = r
        self.log_stop_experiment(self)

    def grid_search(self, parameters=None):
        """
        This function searches a grid of the parameter combinations given into the function
        :param parameters: A nested dictionary, where the outermost key is the estimator name and
        the second level key is the parameter name, and the value is a list of possible parameters
        :return: None
        """
        if parameters is None:
            self.run()
            return

        # Generate every possible combination of the provided hyper parameters.
        workflow = self._workflow
        master_list = []
        params_list = []
        self.log_start_experiment(self)
        for estimator in parameters:
            param_dict = parameters.get(estimator)
            for params in param_dict:
                master_list.append(param_dict.get(params))
                params_list.append(''.join([estimator, '.', params]))
        grid = itertools.product(*master_list)

        # For each tuple in the combination create a run
        for element in grid:
            run_name = ''
            prev_estimator = ''
            # Get all the parameters to be used on set_param
            for param, idx in zip(params_list, range(0, len(params_list))):
                split_params = param.split(sep='.')
                estimator = workflow._pipeline.named_steps.get(split_params[0])
                estimator.set_params(**{split_params[1]: element[idx]})

                # If a new estimator is found,close the parameters of the previous estimator
                # by adding '];' and add the new estimator name and '['
                if prev_estimator == split_params[0]:
                    run_name = ''.join([run_name, split_params[1][0:4], '(', str(element[idx])[0:4], ')'])
                else:
                    run_name = ''.join(
                        [run_name, '];', split_params[0], '[', split_params[1][0:4], '(', str(element[idx])[0:4], ')'])
                    prev_estimator = split_params[0]

            # Remove the initial closing '];' and add the final closing ']'
            run_name = run_name[2:] + ']'
            print(run_name)
            self._metadata['run_id'] = run_name
            r = Run(self, workflow, **dict(self._metadata))
            r.do_splits()
            if self._keep_runs:
                self._runs.append(r)
            self._last_run = r
        self.log_stop_experiment(self)

    @property
    def runs(self):
        if self._keep_runs:
            return self._runs
        else:
            # load splits from backend.
            raise NotImplemented()

    @property
    def last_run(self):
        return self._last_run

    def __str__(self):
        s = []
        if self.id is not None:
            s.append("id:" + str(self.id))
        if self.name is not None and self.name != self.id:
            s.append("name:" + str(self.name))
        if len(s) == 0:
            return str(super())
        else:
            return "Experiment<" + ";".join(s) + ">"

    def traverse_dict(self, dictionary=None):
        # This function traverses a Nested dictionary structure such as the
        # parameter dictionary obtained from hyperparameters()
        # The aim of this function is to convert the param objects to
        # JSON serializable form. The <class 'padre.visitors.parameter.Parameter'> type
        # is used to store the base values. This function changes the type to basic JSON
        # serializable data types.

        if dictionary is None:
            return

        for key in dictionary:
            if isinstance(dictionary[key], padre.visitors.parameter.Parameter):
                dictionary[key] = {'value': dictionary[key].value,
                                   'attributes': dictionary[key].attributes}

            elif isinstance(dictionary[key], dict):
                self.traverse_dict(dictionary[key])

        return dictionary
