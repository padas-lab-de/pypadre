from pypadre.core.base import MetadataMixin, ChildMixin
from pypadre.core.model.generic.i_model_mixins import StoreableMixin
from pypadre.core.printing.tablefyable import Tablefyable


class Split(StoreableMixin, MetadataMixin, ChildMixin, Tablefyable):
    """
    A split is a single part of a execution and the actual excution over parts of the dataset.
    According to the experiment setup the pipeline/workflow will be executed
    """

    def __init__(self, run, num, train_idx, val_idx, test_idx, **kwargs):
        self._num = num
        self._train_idx = train_idx
        self._val_idx = val_idx
        self._test_idx = test_idx
        self._keep_splits = kwargs.pop("keep_splits", False)
        self._splits = []
        self._id = kwargs.pop("split_id", None)
        self._run = run
        # Add defaults
        defaults = {}

        # Merge defaults
        metadata = {**defaults, **kwargs.pop("metadata", {})}
        super().__init__(schema_resource_name="split.json", metadata=metadata, parent=run, **kwargs)

    @property
    def execution(self):
        return self.parent

    @property
    def number(self):
        return self._num

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
        return self.execution.experiment.dataset

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
