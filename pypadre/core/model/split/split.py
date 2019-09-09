import uuid

import numpy as np

from pypadre.pod.base import MetadataEntity, exp_events, phases
from pypadre.core.model.split.custom_split import split_obj
from pypadre.pod.eventhandler import trigger_event, assert_condition


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
        self._strategy = options.pop("strategy", "random")

        assert_condition(condition=self._strategy in ['random', 'cv', 'function', 'index', None],
                         source=self,
                         message=f"Unknown splitting strategy {self._strategy}. "
                         f"Only 'cv', 'random', 'function' or 'None'  allowed")

        self._test_ratio = options.pop("test_ratio", 0.25)
        trigger_event('EVENT_WARN', condition=self._test_ratio is None or (0.0 <= self._test_ratio <= 1.0),
                      source=self,
                      message=f"Wrong ratio of test set provided {self._test_ratio}. Continuing with default=0")
        self._val_ratio = options.pop("val_ratio", 0)
        trigger_event('EVENT_WARN', condition=self._val_ratio is None or (0.0 <= self._val_ratio <= 1.0),
                      source=self,
                      message=f"Wrong ratio of evaluation set provided {self._val_ratio}. Continuing with default=0")
        self._n_folds = options.pop("n_folds", 3)
        assert_condition(condition=1 <= self._n_folds, source=self, message=f"Number of folds not positive {self._n_folds}")
        self._random_seed = options.pop("random_seed", None)
        self._no_shuffle = options.pop("no_shuffle", False)
        trigger_event('EVENT_WARN',
                      condition=not (self._n_folds == 1 and self._strategy == "random" and self._no_shuffle),
                      source=self,
                      message=f"Random test split will be always the same since shuffling is not permitted")
        assert_condition(condition=self._n_folds < self._dataset.size[0] or self._strategy != "cv",
                         source=self,
                         message=f"There are more folds than examples: {self._n_folds}<{self._dataset.size[0]}")
        self._stratified = options.pop("stratified", None)
        self._indices = options.pop("indices", None)
        if self._strategy == "indices":
            assert_condition(condition=self._indices is not None, source=self,
                             message=f"Splitting strategy {self._strategy} requires an explicit split given by parameter 'indices'")
        if self._stratified is None:
            self._stratified = ds.targets() is not None
        else:
            if self._stratified and ds.targets() is None:
                trigger_event('EVENT_WARN',
                              condition=False,
                              source=self,
                              message=f"Targets not provided in dataset {ds}. Can not do stratified splitting")
                self._stratified = False
        self._splitting_fn = options.pop("fn", None)
        if self._strategy == "function":
            assert_condition(condition=split_obj.function_pointer is not None, source=self,
                             message=f"Splitting strategy {self._strategy} requires a function provided via parameter 'fn'")

        self._index_list = options.pop('index', None)

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
                train, test, val = split_obj.function_pointer(idx)
                yield train, test, val
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
                    # The test array can be seen as a non overlapping sub array of size n_te moving from start to end
                    n_te = i * int(n / self._n_folds)
                    test = np.asarray(range(n_te, n_te + int(n / self._n_folds)))

                    # if the test array exceeds the end of the array wrap it around the beginning of the array
                    test = np.mod(test, n)

                    # The training array is the set difference of the complete array and the testing array
                    train = np.asarray(list(set(idx) - set(test)))

                    if self._val_ratio > 0:  # create a validation set out of the test set
                        n_v = int(len(train) * self._val_ratio)
                        yield train[:n_v], test, train[n_v:]
                    else:
                        yield train, test, None

            elif self._strategy == "index":
                # If a list of dictionaries are given to the experiment as indices, pop each one out and return
                for i in range(len(self._index_list)):
                    train = self._index_list[i].get('train', None)
                    if train is not None:
                        train = np.array(train)

                    test = self._index_list[i].get('test', None)
                    if test is not None:
                        test=np.array(test)

                    val = self._index_list[i].get('val', None)
                    if val is not None:
                        val = np.array(val)
                        
                    yield train, test, val

            else:
                raise ValueError(f"Unknown splitting strategy {self._splitting_strategy}")

        return splitting_iterator()


class Split(MetadataEntity):
    """
    A split is a single part of a run and the actual excution over parts of the dataset.
    According to the experiment setup the pipeline/workflow will be executed
    """

    def __init__(self, run, num, train_idx, val_idx, test_idx, **options):
        self._run = run
        self._num = num
        #self._backend = run.backend
        #self._stdout = run.stdout
        self._train_idx = train_idx
        self._val_idx = val_idx
        self._test_idx = test_idx
        self._keep_splits = options.pop("keep_splits", False)
        self._splits = []
        self._id = options.pop("split_id", None)
        super().__init__(**options)

        if self._id is None:
            self._id = uuid.uuid4()

    @property
    def number(self):
        return self._num

    @property
    def run(self):
        return self._run

    def execute(self):
        # Fire event
        trigger_event('EVENT_START_SPLIT', split=self)

        # log run start here.
        workflow = self._run.execution.experiment.workflow

        # Fire event
        trigger_event('EVENT_LOG_EVENT',
                      source=self, kind=exp_events.start, phase=phases.fitting)

        workflow.fit(self)
        trigger_event('EVENT_LOG_EVENT',
                      source=self, kind=exp_events.stop, phase=phases.fitting)
        if workflow.is_inferencer() and self.has_testset():
            trigger_event('EVENT_LOG_EVENT',
                          source=self, kind=exp_events.start, phase=phases.inferencing)
            workflow.infer(self, self.train_idx.tolist(), self.test_idx.tolist())
            trigger_event('EVENT_LOG_EVENT',
                          source=self, kind=exp_events.stop, phase=phases.inferencing)
        # Fire event
        trigger_event('EVENT_STOP_SPLIT', split=self)

    def has_testset(self):
        return self._test_idx is not None and len(self._test_idx) > 0

    def has_valset(self):
        return self._val_idx is not None and len(self._val_idx) > 0

    def has_targets(self):
        return self.dataset.targets() is not None and len(self.dataset.targets()) > 0

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
        if self.uid is not None:
            s.append("id:" + str(self.id))
        if self.name is not None and self.name != self.id:
            s.append("name:" + str(self.name))
        if len(s) == 0:
            return str(super())
        else:
            return "Split<" + ";".join(s) + ">"
