import numpy as np

from pypadre._package import PACKAGE_ID
from pypadre.core.model.code.code_mixin import Function
from pypadre.core.model.dataset.dataset import Dataset
from pypadre.core.model.split.split import Split
from pypadre.core.util.utils import unpack


def split(ctx, strategy="random", test_ratio=0.25, random_seed=None, val_ratio=0,
          n_folds=3, shuffle=True, stratified=None, indices=None, index=None):
    (data, run, component, predecessor) = unpack(ctx, "data", "run", "component", ("predecessor", None))
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
         - shuffle={True|False} indicates, whether shuffling the data is allowed.
         - indices = [(train, validation, test)] a list of tuples with three index arrays in the dataset.
                                   Every index array contains
                                   the row index of the datapoints contained in the split
         - fn                      function of the form fn(dataset, **options) that returns a iterator over
                                   (train, validation, test) tuples (the form is similar to the indices parameter) as split
        """
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

    if random_seed is None:
        from pypadre.core.util.random import padre_seed
        random_seed = padre_seed
    r = np.random.RandomState(random_seed)
    idx = np.arange(n)

    def splitting_iterator():
        num = -1
        # now apply splitting strategy
        # todo s: time aware cross validation, stratified splits,
        # Todo do sanity checks that indizes do not overlap
        if strategy is None:
            yield Split(run=run, num=++num, train_idx=idx, test_idx=None, val_idx=None, component=component,
                        predecessor=predecessor)
        elif strategy == "explicit":
            for i in indices:
                # TODO FIXME
                yield i
        elif strategy == "random":
            # for i in range(n_folds):
            if shuffle:  # Reshuffle every "fold"
                r.shuffle(idx)
            n_tr = int(n * (1.0 - test_ratio))
            train, test = idx[:n_tr], idx[n_tr:]
            num += 1
            if val_ratio > 0:  # create a validation set out of the test set
                n_v = int(len(train) * val_ratio)
                yield Split(run=run, num=num, train_idx=train[:n_v], test_idx=test, val_idx=train[n_v:],
                            component=component, predecessor=predecessor)
            else:
                yield Split(run=run, num=num, train_idx=train, test_idx=test, val_idx=None, component=component,
                            predecessor=predecessor)
        elif strategy == "cv":
            for i in range(n_folds):
                # The test array can be seen as a non overlapping sub array of size n_te moving from start to end
                n_te = i * int(n / n_folds)
                test = np.asarray(range(n_te, n_te + int(n / n_folds)))

                # if the test array exceeds the end of the array wrap it around the beginning of the array
                test = np.mod(test, n)

                # The training array is the set difference of the complete array and the testing array
                train = np.asarray(list(set(idx) - set(test)))

                if val_ratio > 0:  # create a validation set out of the test set
                    n_v = int(len(train) * val_ratio)
                    yield Split(run=run, num=++num, train_idx=train[:n_v], test_idx=test, val_idx=train[n_v:],
                                component=component, predecessor=predecessor)
                else:
                    yield Split(run=run, num=++num, train_idx=train, test_idx=test, val_idx=None, component=component,
                                predecessor=predecessor)

        elif strategy == "index":
            # If a list of dictionaries are given to the experiment as indices, pop each one out and return
            for i in range(len(index)):
                train = index[i].get('train', None)
                if train is not None:
                    train = np.array(train)

                test = index[i].get('test', None)
                if test is not None:
                    test = np.array(test)

                val = index[i].get('val', None)
                if val is not None:
                    val = np.array(val)

                yield Split(run=run, num=++num, train_idx=train, test_idx=test, val_idx=val, component=component,
                            predecessor=predecessor)

        else:
            raise ValueError(f"Unknown splitting strategy {strategy}")

    return splitting_iterator()


default_split = Function(fn=split, transient=True,
                         identifier=PACKAGE_ID)
