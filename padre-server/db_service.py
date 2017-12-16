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
                DataSetService().save(i.name, i)
        for post in Dataset.objects:
            print(post.name)
        logger.debug("init_setup : END")

def setup(self):
    pass

class DataSetService(object):
    def save(self, dataset_name, dataset):
        dataset_obj = Dataset(name=dataset_name)
        dataset_obj.metadata.put(JSonSerializer.serialise(dataset.metadata), encoding="UTF-16",
                             filename=dataset_name + "_metadata")
        dataset_obj.data.put(PickleSerializer.serialise(dataset.data), encoding="UTF-16", filename=dataset_name + "_data")
        dataset_obj.target.put(PickleSerializer.serialise(dataset.target), encoding="UTF-16", filename=dataset_name + "_target")
        dataset_obj.save()

    def get(self, dataset_name):
        dataset = [data.name for data in Dataset.objects(name=dataset_name)]
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