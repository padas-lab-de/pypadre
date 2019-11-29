import networkx
import numpy as np

from pypadre.core.ontology.padre_ontology import PaDREOntology
from pypadre.pod.importing.dataset.dataset_import import SnapLoader, KonectLoader, SKLearnLoader, OpenMLLoader
from pypadre.pod.tests.base_test import PadreAppTest


class Test_loaders(PadreAppTest):
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
        if ds.data().is_directed():
            assert ds.metadata["type"] == PaDREOntology.SubClassesDataset.Graph.value + "Directed"
        else:
            assert ds.metadata["type"] == PaDREOntology.SubClassesDataset.Graph.value

    def test_konect_loader(self):

        loader = KonectLoader()

        loader.list()

        ds = loader.load("konect", url= "http://konect.cc/networks/komarix-citeseer")

        assert isinstance(ds.data(), networkx.Graph)
        assert len(ds.attributes) == 2

    def test_openml_loader(self):

        loader = OpenMLLoader()

        ds = loader.load(source="41078")

        assert ds.name == "iris"
        assert ds.attributes[0].name == "sepallength"




