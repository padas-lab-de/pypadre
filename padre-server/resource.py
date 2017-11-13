from flask_restful import Resource
from padre.utils import DefaultLogger, ResourceDirectory
from padre.repository import PadreFileRepository
import os
from flask import jsonify
from werkzeug.exceptions import HTTPException

from authentication import requires_auth


# TODO create a corresponding configuration object. look up best practices
# data_dir = os.path.expanduser("~/tmp/padre_srv/data")
# if not os.path.exists(data_dir):
#     os.makedirs(data_dir)
repo = PadreFileRepository(ResourceDirectory().create_directory())



# Register a default logger
logger = DefaultLogger().get_default_logger()
# class Resource:
#
#     def __init__(self, app):
#         self.app = app

class DatasetsAPI(Resource):  # Create a RESTful resource for datasets

    @requires_auth
    def get(self):  # Create GET endpoint
        logger.debug("start")
        logger.debug("end")
        HTTPException.code = 400
        raise HTTPException
        return jsonify(data=repo.list())
        # return "get"

    def put(self):  # update a new dataset
        pass

    def post(self):  # update a new dataset
        pass

    def delete(self):  # delete a new dataset
        pass


class DatasetAPI(Resource):  # Create a RESTful resource for datasets

    def get(self, name):  # Create GET endpoint
        print(name)
        return repo.get(name).metadata

    def put(self):  # update a new dataset
        pass

    def post(self):  # update a new dataset
        pass

    def delete(self):  # delete a new dataset
        pass
