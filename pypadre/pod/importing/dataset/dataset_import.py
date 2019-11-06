import inspect
import os
import re
from abc import abstractmethod, ABCMeta

import networkx as nx
import numpy as np
import pandas as pd
import sklearn.datasets as ds
from padre.PaDREOntology import PaDREOntology

# import openml as oml
from pypadre.core.model.dataset.attribute import Attribute
from pypadre.core.model.dataset.dataset import Dataset
from pypadre.core.model.generic.i_model_mixins import LoggableMixin
from pypadre.core.util.utils import _Const
from pypadre.pod.importing.dataset.graph_import import create_from_snap, create_from_konect


class _Sources(_Const):
    file = "file"
    openml = "openml"
    graph = "graph"


sources = _Sources()


class DataSetLoaderMixin(LoggableMixin):
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

    def _create_dataset(self, **kwargs):
        # TODO extract attributes
        dataset = Dataset(metadata=kwargs)
        self.send_warn(condition=len(dataset.metadata["targets"]) == 0, source=self,
                       message='No targets defined. Program will crash when used for supervised learning')
        return dataset


class ICollectionDataSetLoader(DataSetLoaderMixin):
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


class CSVLoader(DataSetLoaderMixin):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
        self.send_error(message="Dataset path does not exist", condition=os.path.exists(os.path.abspath(source)),
                        source=self)

        # TODO something else than multivariat?
        meta = {**{"name": source.split('/')[-1].split('.csv')[0],
                   "description": "imported from csv", "source": source}, **kwargs}
        data_set = self._create_dataset(**meta)

        # TODO maybe check this somewhere else?
        data = pd.read_csv(source)
        self.send_warn(condition=data.applymap(np.isreal).all(1).all() == True,
                       source=self.__class__.__name__ + inspect.currentframe().f_code.co_name,
                       message='Non-numeric data values found. Program may crash if not handled by estimators')

        # find targets by searching in the string
        targets = meta.get("targets")
        for col_name in targets:
            data[col_name] = data[col_name].astype('category')
            data[col_name] = data[col_name].cat.codes

        # Extract attributes from column names #TODO for now it is done inside the container
        # atts = []
        # for feature in data.columns.values:
        #     atts.append(Attribute(name=feature,
        #                           measurementLevel="Ratio" if feature in targets else None,
        #                           defaultTargetAttribute=feature in targets))
        data_set.set_data(data)
        return data_set


class PandasLoader(DataSetLoaderMixin):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

        # if len(meta["targets"]) == 0:
        #     meta["targets"] = [0] * len(source)
        data_set = self._create_dataset(**meta)

        data_set.set_data(source)
        return data_set


class NumpyLoader(DataSetLoaderMixin):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
        atts = []
        for feature in columns:
            atts.append(Attribute(name=feature, measurementLevel=None, unit=None, description=None,
                                  defaultTargetAttribute=feature in target_features, context=None))

        meta = {**{"name": "numpy_imported",
                   "description": "imported by numpy multidimensional",
                   "originalSource": "https://imported/from/numpy/NumpyArray.html", "attributes": atts,
                   "targets": target_features}, **kwargs}

        data_set = self._create_dataset(**meta)

        data_set.set_data(source)
        # FIXME add none multi dim data
        # FIXME add multidimensional data
        return data_set


class NetworkXLoader(DataSetLoaderMixin):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def mapping(source):
        return isinstance(source, nx.Graph)

    def load(self, source, target_features=None, **kwargs):
        """
        Takes a networkx graph object and creates a padre dataset.
        :param source: networkx graph object
        :param target_features: (list) targets attributes names
        :param kwargs: Additional meta data for the created dataset (e.g, attributes)
        :return: A dataset object
        """

        meta = {**{"name": "networkx_imported",
                   "description": "imported by networkx graph",
                   "originalSource": "https://imported/from/networkx/Graph.html",
                   "targets": target_features}, **kwargs}

        data_set = self._create_dataset(**meta)

        data_set.set_data(source)

        return data_set


class SKLearnLoader(ICollectionDataSetLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

    def list(self, **kwargs):
        return ds.__all__

    def load_default(self):
        for loader in self.loaders:
            yield self.load("sklearn", utility=loader[0], type=loader[1][1], originalSource=loader[2])

    @staticmethod
    def mapping(source):
        return source.__eq__("sklearn")

    def load(self, source, utility: str = None, **kwargs):

        if not utility or getattr(ds, utility) is None or not callable(getattr(ds, utility)):
            raise ValueError(
                "A sklearn utility name has to be passed with the utility parameter to specify which data set to load.")

        bunch = getattr(ds, utility)()
        # name = os.path.splitext(os.path.basename(bunch['filename']))[0]
        # description = bunch["DESCR"]

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
                atts.append(Attribute(fn[ix], PaDREOntology.SubClassesMeasurement.Ratio.value,
                                      unit=PaDREOntology.SubClassesUnit.Count.value, description="TODO",
                                      defaultTargetAttribute=n_feat <= ix, index=ix,
                                      type=PaDREOntology.SubClassesDatum.Character.value))
            else:
                atts.append(Attribute(str(ix), PaDREOntology.SubClassesMeasurement.Ratio.value,
                                      unit=PaDREOntology.SubClassesUnit.Count.value, description="TODO",
                                      defaultTargetAttribute=n_feat <= ix, index=ix,
                                      type=PaDREOntology.SubClassesDatum.Character.value))

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def list(self, **kwargs):
        pass

    def load_default(self):
        return []

    @staticmethod
    def mapping(source):
        return source.__eq__("snap")

    def load(self, source, url="", link_num=0, **kwargs):
        """Takes the graph of the Snap website and puts it into a pypadre.dataset.

        Args:
            url (str): The url of the specific graph. From graph of this website: https://snap.stanford.edu/
            link_num (int): Some Graphs have several datasets and thus several download-links.
        Returns:
            A pypadre.dataset object that conatins the graph of the url.
        """
        graph, meta = create_from_snap(url, link_num=link_num, logger=self)

        meta = {**meta, **kwargs}

        data_set = self._create_dataset(**meta)
        data_set.set_data(graph)

        return data_set


class KonectLoader(ICollectionDataSetLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def list(self, **kwargs):
        pass

    def load_default(self):
        return []

    @staticmethod
    def mapping(source):
        return source.__eq__("konect")

    def load(self, source, url="", zero_based=False, **kwargs):
        graph, meta = create_from_konect(url=url, zero_based=zero_based)

        meta = {**meta, **kwargs}

        data_set = self._create_dataset(**meta)
        data_set.set_data(graph)

        return data_set

# class OpenMlLoader(ICollectionDataSetLoader):
#     DATATYPE_MAP = {'INTEGER': (PaDREOntology.SubClassesDatum.Integer.value,PaDREOntology.SubClassesMeasurement.Ratio.value),
#                     'NUMERIC': (PaDREOntology.SubClassesDatum.Number.value, PaDREOntology.SubClassesMeasurement.Ratio.value),
#                     'REAL': (PaDREOntology.SubClassesDatum.Float.value,PaDREOntology.SubClassesMeasurement.Ratio.value) ,
#                     'STRING': (PaDREOntology.SubClassesDatum.Character.value,PaDREOntology.SubClassesMeasurement.Nominal)}
#
#     def list(self, **kwargs):
#         # TODO return openMl datasets
#         pass
#
#     def load_default(self):
#         # TODO load some default datasets
#         return []
#
#     @staticmethod
#     def mapping(source):
#         return source.__eq__("openml")
#
#     def load(self, source, url="",**kwargs):
#         """
#
#         :param source:
#         :param url:
#         :param kwargs:
#         :return:
#         """
#
#         dataset_id = url.split("/")[-1].strip(" ")
#         # oml.config.apikey = apikey
#         with tempfile.TemporaryDirectory(suffix=dataset_id) as temp_dir:
#             oml.config.cache_directory = temp_dir
#
#             try:
#                 load = oml.datasets.get_dataset(int(dataset_id))
#             except exceptions.OpenMLServerException as err:
#                 print("Dataset not found! \nErrormessage: " + str(err))
#                 return None
#             except ConnectionError as err:
#                 print("openML unreachable! \nErrormessage: " + str(err))
#                 return None
#             except OSError as err:
#                 print("Invalid datapath! \nErrormessage: " + str(err))
#                 return None
#
#             meta = {**{"name": load.name, "version": load.version, "description": load.description,
#                     "originalSource":load.url,"type": PaDREOntology.SubClassesDataset.Multivariat.value},**kwargs}
#
#             temp_file = open(temp_dir+'/org/openml/www/datasets/'+dataset_id+'/dataset.arff', encoding='utf-8')
#             bunch = arff.load(temp_file)
#             temp_file.close()
#             bunch_atts = bunch["attributes"]
#             attributes = []
#
#             for att in bunch_atts:
#                 attributes.append(att[0])
#
#             df_data = pd.DataFrame(data = bunch['data'])
#             atts = []
#
#             for col in df_data.columns:
#                 unit = None
#                 measurment_lvl = None
#                 data_type = None
#                 curr_attribute = bunch_atts[col]
#                 self.send_error(message="Name failure, Inconsistency encountered in attributes names",
#                                 condition=load.features[col]!=curr_attribute[0])
#
#                 if isinstance(curr_attribute[1], list):
#                     measurment_lvl = PaDREOntology.SubClassesMeasurement.Nominal.value
#                     if any( op in ''.join(curr_attribute[1]) for op in ["<",">","="]):
#                         measurment_lvl = PaDREOntology.SubClassesMeasurement.Interval.value
#                     data_type = PaDREOntology.SubClassesDatum.Character.value
#                     df_data[col] = df_data[col].astype('category')
#                 elif isinstance(curr_attribute[1], str) and curr_attribute[1] in self.DATATYPE_MAP.keys():
#                     data_type = self.DATATYPE_MAP[curr_attribute[1]][0]
#                     measurment_lvl = self.DATATYPE_MAP[curr_attribute[1]][1]
#                 else:
#                     self.send_error(message="Invalid data format from openml")
#
#                 atts.append(Attribute(name=curr_attribute[0],measurementLevel=measurment_lvl, unit=unit, type=data_type,
#                                       defaultTargetAttribute=(curr_attribute[0] == load.default_target_attribute)))
#
#             df_data.columns = attributes
#             meta["attributes"] = atts
#             dataset = self._create_dataset(**meta)
#
#             dataset.set_data(df_data)
#
#             return dataset
