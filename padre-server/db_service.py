from mongoengine import connect
from mongoengine.connection import _get_db
from model import Dataset
from padre.utils import DefaultLogger, ResourceDirectory
from padre.constants import DEBUG
from padre.repository import PadreFileRepository, PickleSerializer, JSonSerializer
from padre.ds_import import load_sklearn_toys

repo = PadreFileRepository(ResourceDirectory().create_directory())

logger = DefaultLogger.get_default_logger()


class Setup(object):

    @staticmethod
    def connect_database():
        logger.debug("Trying to connect to database")
        connect('test_db')
        logger.debug("Connected successfully to database")

    @staticmethod
    def close_setup():
        logger.debug("close_setup : START")
        connect('test_db').drop_database('test_db')
        logger.debug("close_setup : END")

    @staticmethod
    def init_setup():
        logger.debug("init_setup : START")
        Setup.connect_database()
        if DEBUG:
            for i in load_sklearn_toys():
                repo.put(i.name, i)
                dataset = Dataset(name=i.name)
                dataset.metadata.put(JSonSerializer.serialise(i.metadata), encoding="UTF-16",
                                     filename=i.name + "_metadata")
                dataset.data.put(PickleSerializer.serialise(i.data), encoding="UTF-16", filename=i.name + "_data")
                dataset.target.put(PickleSerializer.serialise(i.target), encoding="UTF-16", filename=i.name + "_target")
                dataset.save()
        for post in Dataset.objects:
            print(post.name)
        logger.debug("init_setup : END")

def setup(self):
    pass

class DataSetService(object):
    def save(self, dataset_name, dataset):
        dataset = Dataset(name=dataset_name)
        dataset.metadata.put(JSonSerializer.serialise(dataset.metadata), encoding="UTF-16",
                             filename=dataset.name + "_metadata")
        dataset.data.put(PickleSerializer.serialise(dataset.data), encoding="UTF-16", filename=dataset.name + "_data")
        dataset.target.put(PickleSerializer.serialise(dataset.target), encoding="UTF-16", filename=dataset.name + "_target")
        dataset.save()
        pass

    def get(self, dataset_name):
        dataset = Dataset.objects(name=dataset_name)
        return dataset

    def getAll(self):
        datasets = [data.name for data in Dataset.objects]
        return datasets

    def delete(self):
        pass
    def deleteAll(self):
        pass
    def update(self):
        pass
    def updateAll(self):
        pass