import inspect
import os
import re
import uuid
from abc import abstractmethod, ABCMeta

import networkx as nx
import numpy as np
import pandas as pd
import sklearn.datasets as ds
from padre.PaDREOntology import PaDREOntology

from pypadre.core.model.dataset.attribute import Attribute
from pypadre.core.model.dataset.dataset import Dataset
from pypadre.core.events import assert_condition, trigger_event
from pypadre.core.util.utils import _Const


class _Sources(_Const):
    file = "file"
    openml = "openml"
    graph = "graph"


sources = _Sources()


class IDataSetLoader:
    """
    Class used to load external datasets
    """
    __metaclass__ = ABCMeta

    @staticmethod
    @abstractmethod
    def mapping(source):
        """
        Check if loader should be used by parsing source.
        :return: True if loader should be used
        """
        return False

    @abstractmethod
    def load(self, path, **kwargs):
        """
        This function should download a externally provided dataset
        :param path: path to the file or the download endpoint
        :param kwargs: parameters needed for a download
        :return: Dataset
        """
        raise NotImplementedError()

    @staticmethod
    def _create_dataset(**kwargs):
        meta = {**{"id": str(uuid.uuid4()), "name": "something",
                   "description": "", "version": "1.0",
                   "type": "multivariate", "targets": [], "published": False}, **kwargs}

        trigger_event('EVENT_WARN', condition=len(meta["targets"]) > 0,
                      source=inspect.currentframe().f_code.co_name,
                      message='No targets defined. Program will crash when used for supervised learning')

        # TODO extract attributes
        return Dataset(**meta)


class ICollectionDataSetLoader(IDataSetLoader):
    @abstractmethod
    def list(self, **kwargs):
        """
        This function should list datasets
        :param self:
        :param kwargs:
        :return:
        """
        pass

    @abstractmethod
    def load_default(self):
        pass


class CSVLoader(IDataSetLoader):

    def __hash__(self):
        return hash(self.__class__)

    @staticmethod
    def mapping(source):
        return str.startswith("/") or str.startswith(".") and str.endswith(".csv")

    def load(self, source, **kwargs):
        """Takes the path of a csv file and a list of the target columns and creates a padre-Dataset.

        Args:
            source (str): The path of the csv-file
            kwargs : Parameters for loading a csv file. (name(str): Optional name of dataset, description(str): Optional description of the dataset, source(str): original source - should be url, type(str): type of dataset)

        Returns:
            <class 'pypadre.datasets.Dataset'> A dataset containing the data of the .csv file

        """
        assert_condition(condition=os.path.exists(os.path.abspath(source)),
                         source=self.__class__.__name__ + inspect.currentframe().f_code.co_name,
                         message='Dataset path does not exist')

        # TODO something else than multivariat?
        meta = {**{"name": source.split('/')[-1].split('.csv')[0],
                   "description": "imported from csv", "source": source}, **kwargs}
        data_set = self._create_dataset(**meta)

        # TODO maybe check this somewhere else?
        data = pd.read_csv(source)
        trigger_event('EVENT_WARN', condition=data.applymap(np.isreal).all(1).all() == True,
                      source=self.__class__.__name__ + inspect.currentframe().f_code.co_name,
                      message='Non-numeric data values found. Program may crash if not handled by estimators')

        # find targets by searching in the string
        targets = meta.get("targets")
        for col_name in meta.get("targets"):
            data[col_name] = data[col_name].astype('category')
            data[col_name] = data[col_name].cat.codes

        # Extract attributes from column names
        atts = []
        for feature in data.columns.values:
            atts.append(Attribute(name=feature,
                                  measurementLevel="Ratio" if feature in targets else None,
                                  defaultTargetAttribute=feature in targets))
        data_set.set_data(data, atts)
        return data_set


class PandasLoader(IDataSetLoader):

    @staticmethod
    def mapping(source):
        return isinstance(source, pd.DataFrame)

    def load(self, source, target_features=None, **kwargs):
        """
        Takes a pandas dataframe and a list of the names of target columns and creates a padre-Dataset.

        Args:
        :param source: The pandas dataset.
        :param kwargs: targets (list): The column names of the target features of the pandas-file.

        Returns:
            pypadre.Dataset() A dataset containing the data of the .pandas file

        """
        meta = {**{"name": "pandas_imported_df", "description": "imported by pandas_df",
                   "originalSource": "https://imported/from/pandas/Dataframe.html"}, **kwargs}
        data_set = self._create_dataset(**meta)

        atts = []
        if len(meta["targets"]) == 0:
            meta["targets"] = [0] * len(source)

        for feature in source.columns.values:
            atts.append(Attribute(name=feature, measurementLevel=None, unit=None, description=None,
                                  defaultTargetAttribute=feature in target_features, context=None))
        data_set.set_data(source, atts)
        return data_set


class NumpyLoader(IDataSetLoader):

    @staticmethod
    def mapping(source):
        return isinstance(source, np.ndarray)

    def load(self, source, columns=None, target_features=None, **kwargs):
        """
            Takes a multidimensional numpy array and creates a dataset out of it
            :param source: The input n dimensional numpy array
            :param columns: Array of data column names,
            :param target_features: Target features column names
            :param kwargs: Additional meta info (targets: The targets corresponding to every feature)
            :return: A dataset object
            """
        meta = {**{"name": "numpy_imported",
                   "description": "imported by numpy multidimensional",
                   "originalSource": "https://imported/from/pandas/Dataframe.html"}, **kwargs}
        data_set = self._create_dataset(**meta)

        atts = []
        if len(meta["target_features"]) == 0:
            targets = [0] * len(source)

        for feature in columns:
            atts.append(Attribute(name=feature, measurementLevel=None, unit=None, description=None,
                                  defaultTargetAttribute=feature in target_features, context=None))

        # FIXME add none multi dim data
        # FIXME add multidimensional data
        return data_set


class NetworkXLoader(IDataSetLoader):

    @staticmethod
    def mapping(source):
        return isinstance(source, nx.Graph)

    def load(self, **kwargs):
        pass


class SKLearnLoader(ICollectionDataSetLoader):

    def list(self, **kwargs):
        return ds.__all__

    def load_default(self):
        loaders = [("load_boston", ("regression", PaDREOntology.SubClassesDataset.Multivariat.value),
                    "https://scikit-learn.org/stable/modules/generated/"
                    "sklearn.datasets.load_boston.html#sklearn.datasets.load_boston"),
                   ("load_breast_cancer", ("classification", PaDREOntology.SubClassesDataset.Multivariat.value),
                    "https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)"),
                   ("load_diabetes", ("regression", PaDREOntology.SubClassesDataset.Multivariat.value),
                    "https://scikit-learn.org/stable/modules/generated/"
                    "sklearn.datasets.load_diabetes.html#sklearn.datasets.load_diabetes"),
                   ("load_digits", ("classification", PaDREOntology.SubClassesDataset.Multivariat.value),
                    "http://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits"),
                   ("load_iris", ("classification", PaDREOntology.SubClassesDataset.Multivariat.value),
                    "https://scikit-learn.org/stable/modules/generated/"
                    "sklearn.datasets.load_iris.html#sklearn.datasets.load_iris"),
                   ("load_linnerud", ("mregression", PaDREOntology.SubClassesDataset.Multivariat.value),
                    "https://scikit-learn.org/stable/modules/generated/"
                    "sklearn.datasets.load_linnerud.html#sklearn.datasets.load_linnerud")]

        for loader in loaders:
            yield self.load("sklearn", utility=loader[0], type=loader[1][1], originalSource=loader[2])

    @staticmethod
    def mapping(source):
        return source.__eq__("sklearn")

    def load(self, source, utility: str = None, **kwargs):
        if not utility or getattr(ds, utility) is None or not callable(getattr(ds, utility)):
            raise ValueError(
                "A sklearn utility name has to be passed with the utility parameter to specify which data set to load.")

        bunch = getattr(ds, utility)()
        #name = os.path.splitext(os.path.basename(bunch['filename']))[0]
        #description = bunch["DESCR"]

        name, description = self._split_description(bunch['DESCR'])

        meta = {**{"name": name, "description": description,
                   "originalSource": "https://imported/from/pandas/Dataframe.html"}, **kwargs}

        n_feat = bunch.data.shape[1]
        if len(bunch.target.shape) == 1:
            data = np.concatenate([bunch.data[:, :], bunch.target[:, None]], axis=1)
        else:
            data = np.concatenate([bunch.data[:, :], bunch.target[:, :]], axis=1)
        fn = bunch.get("feature_names")
        atts = []
        # TODO index was NONE, unit, datatype? FIXME
        for ix in range(data.shape[1]):
            if fn is not None and len(fn) > ix:
                atts.append(Attribute(fn[ix], PaDREOntology.SubClassesMeasurement.Ratio.value, unit=PaDREOntology.SubClassesUnit.Count.value, description="TODO", defaultTargetAttribute=n_feat <= ix, index=ix, type=PaDREOntology.SubClassesDatum.Character.value))
            else:
                atts.append(Attribute(str(ix), PaDREOntology.SubClassesMeasurement.Ratio.value, unit=PaDREOntology.SubClassesUnit.Count.value, description="TODO", defaultTargetAttribute=n_feat <= ix, index=ix, type=PaDREOntology.SubClassesDatum.Character.value))

        meta["attributes"] = atts
        data_set = self._create_dataset(**meta)
        data_set.set_data(data)
        return data_set

    @staticmethod
    def _split_description(s):
        # TODO get name of the dataset from somewhere else?
        match = re.compile("\.\. (.*):").match(s)
        if match:
            return match.group(1), s
        else:
            return hash(s), s

    # @staticmethod
    # def _split_description(s):
    #     s = s.strip()
    #     k = s.find("\n")
    #     return s[0:k], s[k + 1:]


class SnapLoader(ICollectionDataSetLoader):

    def list(self, **kwargs):
        pass

    def load_default(self):
        return []

    @staticmethod
    def mapping(source):
        return source.__eq__("snap")

    def load(self, **kwargs):
        pass


class KonectLoader(ICollectionDataSetLoader):

    def list(self, **kwargs):
        pass

    def load_default(self):
        return []

    @staticmethod
    def mapping(source):
        return source.__eq__("konect")

    def load(self, **kwargs):
        pass


class OpenMlLoader(ICollectionDataSetLoader):

    def list(self, **kwargs):
        # TODO return openMl datasets
        pass

    def load_default(self):
        # TODO load some default datasets
        return []

    @staticmethod
    def mapping(source):
        return source.__eq__("openml")

    def load(self, **kwargs):
        pass
