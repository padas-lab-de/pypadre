import numpy as np

from pypadre.core.model.dataset.dataset import Dataset
from pypadre.core.model.split.split import Split


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

    def __init__(self, **options):

        # TODO validation
        # self.validate_input_parameters(options)

        # self._dataset = ds
        # self._num_examples = ds.size[0]
        self._strategy = options.pop("strategy", "random")
        self._test_ratio = options.pop("test_ratio", 0.25)
        self._random_seed = options.pop("random_seed", None)
        self._val_ratio = options.pop("val_ratio", 0)
        self._n_folds = options.pop("n_folds", 3)
        self._no_shuffle = options.pop("no_shuffle", False)
        self._stratified = options.pop("stratified", None)
        self._indices = options.pop("indices", None)
        self._splitting_code = options.pop("code", None)

        self._index_list = options.pop('index', None)

    @property
    def splitting_code(self):
        return self.splitting_code

    def splits(self, *, data, run, component, **kwargs):
        """
        returns an generator function over all available splits. Every iterator returns
        a triple (train_idx, test_idx, eval_idx)
        where *_idx is an index array for getting the particular split of the
        training data set (e.g. dataset.data()[train_idx]
        provides the training slice part)
        :return: generator function
        """
        # TODO look at incoming params
        if isinstance(data, Dataset):
            n = data.size[0]

        # TODO FIXME
        # first create index array and random state vector
        # n = data.size

        # TODO FIXME usage?
        # stratified = self._stratified
        # if stratified is None:
        #     stratified = dataset.targets() is not None
        # else:
        #     if stratified and dataset.targets() is None:
        #         stratified = False

        r = np.random.RandomState(self._random_seed)
        idx = np.arange(n)

        def splitting_iterator():
            num = -1
            # now apply splitting strategy
            # todo s: time aware cross validation, stratified splits,
            # Todo do sanity checks that indizes do not overlap
            if self.splitting_code:
                yield self.splitting_code.call(data, run, ++num, idx, component=component)
            if self._strategy is None:
                yield Split(run, ++num, idx, None, None, component=component)
            elif self._strategy == "explicit":
                for i in self._indices:
                    # TODO FIXME
                    yield i
            elif self._strategy == "random":
                # for i in range(self._n_folds):
                if not self._no_shuffle:  # Reshuffle every "fold"
                    r.shuffle(idx)
                n_tr = int(n * (1.0 - self._test_ratio))
                train, test = idx[:n_tr], idx[n_tr:]
                if self._val_ratio > 0:  # create a validation set out of the test set
                    n_v = int(len(train) * self._val_ratio)
                    yield Split(run=run, num=++num, train_idx=train[:n_v], test_idx=test, val_idx=train[n_v:], component=component)
                else:
                    yield Split(run=run, num=++num, train_idx=train, test_idx=test, val_idx=None, component=component)
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
                        yield Split(run, ++num, train[:n_v], test, train[n_v:], component=component)
                    else:
                        yield Split(run, ++num, train, test, None, component=component)

            elif self._strategy == "index":
                # If a list of dictionaries are given to the experiment as indices, pop each one out and return
                for i in range(len(self._index_list)):
                    train = self._index_list[i].get('train', None)
                    if train is not None:
                        train = np.array(train)

                    test = self._index_list[i].get('test', None)
                    if test is not None:
                        test = np.array(test)

                    val = self._index_list[i].get('val', None)
                    if val is not None:
                        val = np.array(val)

                    yield Split(run, ++num, train, test, val, component=component)

            else:
                raise ValueError(f"Unknown splitting strategy {self._strategy}")

        return splitting_iterator()

    # def validate_input_parameters(self, ds, options):
    #     """
    #     This function validates all the input parameters given to the init function
    #     :param ds: dataset
    #     :param options: Parameters to the split
    #     :return: None, throws an exception if validation fails
    #     """
    #
    #     # Checking for valid splitting strategies
    #     strategy = options.get("strategy", "random")
    #     assert_condition(condition=strategy in ['random', 'cv', 'function', 'index', None],
    #                      source=self,
    #                      message=f"Unknown splitting strategy {strategy}. "
    #                      f"Only 'cv', 'random', 'function' or 'None'  allowed")
    #
    #     # Check whether test ratio is valid
    #     test_ratio = options.get("test_ratio", 0.25)
    #     trigger_event('EVENT_WARN', condition=test_ratio is None or (0.0 <= test_ratio <= 1.0),
    #                   source=self,
    #                   message=f"Wrong ratio of test set provided {test_ratio}. Continuing with default=0")
    #     val_ratio = options.get("val_ratio", 0)
    #     trigger_event('EVENT_WARN', condition=val_ratio is None or (0.0 <= val_ratio <= 1.0),
    #                   source=self,
    #                   message=f"Wrong ratio of evaluation set provided {val_ratio}. Continuing with default=0")
    #
    #     # Check whether the number of folds is positive
    #     n_folds = options.get("n_folds", 3)
    #     assert_condition(condition=1 <= n_folds,
    #                      source=self, message=f"Number of folds not positive {n_folds}")
    #     no_shuffle = options.get("no_shuffle", False)
    #     trigger_event('EVENT_WARN',
    #                   condition=not (n_folds == 1 and strategy == "random" and no_shuffle),
    #                   source=self,
    #                   message=f"Random test split will be always the same since shuffling is not permitted")
    #     assert_condition(condition=n_folds < ds.size[0] or strategy != "cv",
    #                      source=self,
    #                      message=f"There are more folds than examples: {n_folds}<{ds.size[0]}")
    #
    #     # Check if the indices are present if the splitting strategy is indices
    #     if strategy == "indices":
    #         indices = options.get("indices", None)
    #         assert_condition(condition=indices is not None, source=self,
    #                          message=f"Splitting strategy {strategy} requires an explicit split given by "
    #                          f"parameter 'indices'")
    #
    #     stratified = options.get("stratified", None)
    #     if stratified and ds.targets() is None:
    #         trigger_event('EVENT_WARN',
    #                       condition=False,
    #                       source=self,
    #                       message=f"Targets not provided in dataset {ds}. Can not do stratified splitting")
    #
    #     # Check whether the splitting function is given if the splitting strategy is function
    #     splitting_fn = options.get("fn", None)
    #     if strategy == "function":
    #         assert_condition(condition=split_obj.function_pointer is not None, source=self,
    #                          message=f"Splitting strategy {strategy} requires a function "
    #                          f"provided via parameter 'fn'")
    #
    #     # Check if random seed is an integer
    #     random_seed = options.get('random_seed', 0)
    #     assert_condition(condition=isinstance(random_seed, int), source=self,
    #                      message='Random seed should be an integer type')