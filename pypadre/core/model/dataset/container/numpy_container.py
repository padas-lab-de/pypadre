import numpy as np
import pandas as pd
import pandas_profiling as pd_pf
from scipy import stats

from pypadre.core.model.dataset import dataset
from pypadre.core.model.dataset.attribute import Attribute
from pypadre.core.model.dataset.container.base_container import IBaseContainer
from pypadre.core.model.dataset.container.pandas_container import PandasContainer


class NumpyContainer(IBaseContainer):

    def __init__(self, data, attributes=None):
        # todo rework binary data into delegate pattern.
        super().__init__(dataset.formats.numpy, data, attributes)
        self._shape = data.shape
        if attributes is None:
            self._attributes = [Attribute(i, "RATIO") for i in range(data.shape[1])]
            self._data = data
            self._targets_idx = self._shape[1] - 1
            self._features_idx = np.arange(self._shape[1] - 1)
        else:
            if len(attributes) != data.shape[1]:
                raise ValueError("Incorrect number of attributes."
                                 " Data has %d columns, provided attributes %d."
                                 % (data.shape[1], len(attributes)))
            self._data = data
            self._attributes = attributes
            self._targets_idx = np.array([idx for idx, a in enumerate(attributes) if a.defaultTargetAttribute])
            self._features_idx = np.array([idx for idx, a in enumerate(attributes) if not a.defaultTargetAttribute])
            assert set(self._features_idx).isdisjoint(set(self._targets_idx)) and \
                   set(self._features_idx).union(set(self._targets_idx)) == set([idx for idx in range(len(attributes))])

    @property
    def attributes(self):
        return self._attributes

    @property
    def features(self):
        if self._features_idx is None:
            return self._data
        else:
            return self._data[:, self._features_idx]

    @property
    def targets(self):
        if self._targets_idx is None:
            return None
        else:
            return self._data[:, self._targets_idx]

    @property
    def data(self):
        return self._data

    @property
    def shape(self):
        return self._shape

    @property
    def num_attributes(self):
        if self._attributes is None:
            return 0
        else:
            return len(self._attributes)

    def convert(self, bin_format):
        if bin_format is _Formats.pandas:
            # TODO attributes?
            return PandasContainer(pd.DataFrame(self.data))
        return None

    def profile(self, **kwargs):
        return pd_pf.ProfileReport(self.convert(_Formats.pandas).data, **kwargs)

    def describe(self):
        ret = {"n_att": len(self._attributes),
               "n_target": len([a for a in self._attributes if a.defaultTargetAttribute])}
        if self._data is not None:
            ret["stats"] = stats.describe(self._data, axis=0)
        return ret


class NumpyContainerMultiDimensional(NumpyContainer):

    # TODO find a good representation for multidimensional data (do we even need a second container for that??? What about other formats than numpy? Can we convert between some of them?)
    def __init__(self, data, targets):
        super().__init__(_Formats.numpyMulti, data)
        self._shape = data.shape
        if attributes is None:
            self._attributes = [Attribute(i, "RATIO") for i in range(len(attributes))]
            self._data = data
            self._targets_idx = None
            self._features_idx = np.arange(self._shape[1])
        else:

            self._data = data
            self._targets = targets
            self._attributes = attributes
            self._targets_idx = np.array([idx for idx, a in enumerate(attributes) if a.defaultTargetAttribute])
            self._features_idx = np.array([idx for idx, a in enumerate(attributes) if not a.defaultTargetAttribute])
            assert set(self._features_idx).isdisjoint(set(self._targets_idx)) and \
                   set(self._features_idx).union(set(self._targets_idx)) == set([idx for idx in range(len(attributes))])

    @property
    def targets(self):
        return self._targets

    @property
    def features(self):
        return self._data
