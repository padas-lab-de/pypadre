import networkx

from pypadre.core.tests.padre_test import PadreTest
from pypadre.pod.importing.dataset.dataset_import import SnapLoader, KonectLoader, OpenMlLoader


class Test_loaders(PadreTest):
    def __init__(self, *args, **kwargs):
        super(Test_loaders,self).__init__(*args,**kwargs)


    def test_snap_loader(self):

        loader = SnapLoader()

        loader.list()

        Graph = loader.load("snap",url="https://snap.stanford.edu/data/ego-Facebook.html", link_num=0)

        assert isinstance(Graph.data(), networkx.Graph)

    def test_konect_loader(self):

        loader = KonectLoader()

        loader.list()

        graph_ds = loader.load("konect", url= "http://konect.cc/networks/ucidata-zachary")

    def test_openml_loader(self):

        loader = OpenMlLoader()



