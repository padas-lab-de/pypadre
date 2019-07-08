import numpy as np
import pandas as pd
import pandas_profiling as pd_pf
from scipy import stats

from pypadre.core.model.dataset.attribute import Attribute


class PandasContainer:

    def __init__(self, data, attributes=None):
        # todo rework binary data into delegate pattern.
        self._shape = data.shape
        #pd.DataFrame
        if attributes is None:
            self._attributes = [Attribute(i, "RATIO") for i in range(data.shape[1])]
            self._features = data
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
            #TODO assert rework

    @property
    def attributes(self):
        return self._attributes

    @property
    def features(self):
        if self._attributes is None:
            return self._data
        else:
            removekeys = []
            for att in self._attributes:
                if(att.defaultTargetAttribute):
                    removekeys.append(att.name)
            return self._data.drop(removekeys,axis=1).values

    @property
    def targets(self):
        if self._targets_idx is None or len(self._targets_idx) == 0:
            return None
        else:
            targets=[]
            for col,att in enumerate(self._attributes):
                if (att.defaultTargetAttribute):
                    targets.append(att.name)

            # if no targets are present, return None
            if len(targets) == 0:
                return None

            return self._data[targets].values

    @property
    def data(self):
        return self._data

    @property
    def shape(self):
        return self._shape


    @property
    def num_attributes(self):
        return self._data.shape[1]

    def pandas_repr(self):
        return self.data


  #      ret = {"n_att" : len(self._attributes),
  #             "n_target" : len([a for a in self._attributes if a.is_target])}
  #      if self._data is not None:
  #          ret["stats"] = stats.describe(self._data, axis=0)
  #      return ret

    def profile(self,bins=10,check_correlation=True,correlation_threshold=0.9,
                correlation_overrides=None,check_recoded=False):
        return pd_pf.ProfileReport(self.data,bins=bins,check_correlation=check_correlation,correlation_threshold=correlation_threshold,
                correlation_overrides=correlation_overrides,check_recoded=check_recoded)


    def describe(self):
        ret = {"n_att" : len(self._attributes),
               "n_target" : len([a for a in self._attributes if a.defaultTargetAttribute])}
        shallow_cp=self._data
        for col in shallow_cp:
            if isinstance(shallow_cp[col][0], str):
                shallow_cp[col]=pd.factorize(shallow_cp[col])[0]
                print(col)
        ret["status"] = stats.describe(shallow_cp.values)
        return ret