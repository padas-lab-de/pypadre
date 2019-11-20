import numpy as np
import pandas as pd
import pandas_profiling as pd_pf
from scipy import stats

from pypadre.core.model.dataset import dataset
from pypadre.core.model.dataset.attribute import Attribute
from pypadre.core.model.dataset.container.base_container import IBaseContainer
from pypadre.core.model.generic.i_model_mixins import LoggableMixin
from pypadre.core.ontology.padre_ontology import PaDREOntology


class PandasContainer(IBaseContainer, LoggableMixin):

    def __init__(self, data, attributes=None):
        super().__init__(dataset.formats.pandas, data, attributes)
        self._shape = data.shape
        self._data = data
        self._attributes = self.validate_attributes(attributes)
        self._targets_idx = np.array([idx for idx, a in enumerate(self._attributes) if a.defaultTargetAttribute])
        self._features_idx = np.array([idx for idx, a in enumerate(self._attributes) if not a.defaultTargetAttribute])
        assert set(self._features_idx).isdisjoint(set(self._targets_idx)) and \
               set(self._features_idx).union(set(self._targets_idx)) == set(
            [idx for idx in range(len(self._attributes))])

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
                if att.defaultTargetAttribute:
                    removekeys.append(att.name)
            return self._data.drop(removekeys,axis=1).values

    @property
    def targets(self):
        if self._targets_idx is None or len(self._targets_idx) == 0:
            return None
        else:
            targets=[]
            for col,att in enumerate(self._attributes):
                if att.defaultTargetAttribute:
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
        _attributes = [Attribute(name=feat, measurementLevel=PaDREOntology.SubClassesMeasurement.Ratio.value,
                                 unit=PaDREOntology.SubClassesUnit.Count.value, index=i,
                                 type=PaDREOntology.SubClassesDatum.Character.value,
                                 defaultTargetAttribute=(i == data.shape[1] - 1)) or (feat in targets)
                       for i, feat in enumerate(data.columns.values)]
        return _attributes

    def get_ontology(self):
        """
        Looks through the binary data to determine the corresponding ontology of each attribute property
        :return: a dict()
        """
        #TODO
        pass

    def convert(self, bin_format):
        if bin_format is dataset.formats.numpy:
            from pypadre.core.model.dataset.container.numpy_container import NumpyContainer
            return NumpyContainer(self.data.values)
        return None

    def profile(self, **kwargs):
        return pd_pf.ProfileReport(self.data, **kwargs)

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
