import networkx
import pandas as pd
import numpy as np
from padre.PaDREOntology import PaDREOntology

from pypadre.core.tests.padre_test import PadreTest
from pypadre.pod.importing.dataset.dataset_import import SnapLoader, KonectLoader, SKLearnLoader


class Test_loaders(PadreTest):
    def __init__(self, *args, **kwargs):
        super(Test_loaders,self).__init__(*args,**kwargs)

    def test_sklearn_loader(self):

        loader = SKLearnLoader()

        defaults = loader.load_default()

        ds = loader.load("sklearn", utility= loader.loaders[2][0])

        assert ds.name == "_diabetes_dataset"
        assert isinstance(ds.data(),np.ndarray)

    def test_snap_loader(self):

        loader = SnapLoader()

        loader.list()

        ds = loader.load("snap",url="https://snap.stanford.edu/data/ego-Facebook.html", link_num=0)

        assert isinstance(ds.data(), networkx.Graph)
        assert ds.metadata["type"] == PaDREOntology.SubClassesDataset.Graph.value
        assert ds.name == "ego-Facebook"

        ds = loader.load("snap",url="https://snap.stanford.edu/biodata/datasets/10017/10017-ChChSe-Decagon.html")

        assert isinstance(ds.data(),networkx.Graph)
        assert ds.metadata["type"] == PaDREOntology.SubClassesDataset.Graph.value

    def test_konect_loader(self):

        loader = KonectLoader()

        loader.list()

        ds = loader.load("konect", url= "http://konect.cc/networks/komarix-citeseer")

        assert isinstance(ds.data(), networkx.Graph)
        assert len(ds.attributes) == 2

    # def test_openml_loader(self):
    #
    #     loader = OpenMlLoader()
    #
    #     ds = loader.load("openml", url="https://www.openml.org/d/41078")
    #
    #     assert ds.name == "iris"
    #     assert ds.attributes[0].name == "sepallength"




