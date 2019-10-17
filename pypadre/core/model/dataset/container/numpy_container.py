from typing import List

import numpy as np
import pandas as pd
import pandas_profiling as pd_pf
from padre.PaDREOntology import PaDREOntology
from scipy import stats

from pypadre.core.model.dataset import dataset
from pypadre.core.model.dataset.attribute import Attribute
from pypadre.core.model.dataset.container.base_container import IBaseContainer
from pypadre.core.model.dataset.container.pandas_container import PandasContainer
from pypadre.core.model.generic.i_model_mixins import ILoggable


class NumpyContainer(IBaseContainer,ILoggable):

    def __init__(self, data, attributes: List[Attribute]=None):
        # todo rework binary data into delegate pattern.
        super().__init__(dataset.formats.numpy, data, attributes)
        self._shape = data.shape
        self._data = data
        self._attributes = self.validate_attributes(attributes)
        self._targets_idx = np.array([idx for idx, a in enumerate(self._attributes) if a.defaultTargetAttribute])
        self._features_idx = np.array([idx for idx, a in enumerate(self._attributes) if not a.defaultTargetAttribute])
        assert set(self._features_idx).isdisjoint(set(self._targets_idx)) and \
               set(self._features_idx).union(set(self._targets_idx)) == set([idx for idx in range(len(self._attributes))])


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
        if bin_format is dataset.formats.pandas:
            # TODO attributes?
            return PandasContainer(pd.DataFrame(self.data))
        return None

    def validate_attributes(self, attributes=None):
        # TODO look for validating the attributes properties with regards to the ontology
        if attributes is None or len(attributes) == 0:
            self.send_warn(message='Attributes are missing! Attempting to derive them from the binary...',
                           condition=True)
            attributes = self.derive_attributes(self.data)

        self.send_error(message="Incorrect number of attributes. Data has %d columns, provided attributes %d."
                                % (self.shape[1], len(attributes)), condition=len(attributes) == self.shape[1])
        return attributes

    @staticmethod
    def derive_attributes(data, targets=None):
        if targets is None:
            targets = []
        _attributes = [Attribute(name=str(i), measurementLevel=PaDREOntology.SubClassesMeasurement.Ratio.value,
                                 unit=PaDREOntology.SubClassesUnit.Count.value, index=i,
                                 type=PaDREOntology.SubClassesDatum.Character.value,
                                 defaultTargetAttribute=(i == data.shape[1] - 1) or (i in targets))
                       for i in range(data.shape[1])]
        return _attributes

    def get_ontology(self):
        """
        Looks through the binary data to determine the corresponding ontology of each attribute property
        :return: a dict()
        """
        #TODO
        pass

    def profile(self, **kwargs):
        return pd_pf.ProfileReport(self.convert(dataset.formats.pandas).data, **kwargs)

    def describe(self):
        ret = {"n_att": len(self._attributes),
               "n_target": len([a for a in self._attributes if a.defaultTargetAttribute])}
        if self._data is not None:
            ret["stats"] = stats.describe(self._data, axis=0)
        return ret


class NumpyContainerMultiDimensional(NumpyContainer):

    # TODO find a good representation for multidimensional data (do we even need a second container for that??? What about other formats than numpy? Can we convert between some of them?)
    def __init__(self, data, targets):
        super().__init__(dataset.formats.numpyMulti, data)
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
