import numpy as np
import pandas as pd
import pandas_profiling as pd_pf
from scipy import stats

from pypadre.core.model.dataset.attribute import Attribute


class NumpyContainer:

    def __init__(self, data, attributes=None):
        # todo rework binary data into delegate pattern.
        self._shape = data.shape
        if attributes is None:
            self._attributes = [Attribute(i, "RATIO") for i in range(data.shape[1])]
            self._data = data
            self._targets_idx = None
            self._features_idx = np.arange(self._shape[1])
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

    def pandas_repr(self):
        return pd.DataFrame(self.data)



    def profile(self,bins=10,check_correlation=True,correlation_threshold=0.9,
                correlation_overrides=None,check_recoded=False):
        return pd_pf.ProfileReport(pd.DataFrame(self.data), bins=bins, check_correlation=check_correlation,
                                   correlation_threshold=correlation_threshold,
                                   correlation_overrides=correlation_overrides,check_recoded=check_recoded)


    def describe(self):
        ret = {"n_att": len(self._attributes),
               "n_target": len([a for a in self._attributes if a.defaultTargetAttribute])}
        if self._data is not None:
            ret["stats"] = stats.describe(self._data, axis=0)
        return ret


class NumpyContainerMultiDimensional(NumpyContainer):

    def __init__(self, data, targets, attributes=None):
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