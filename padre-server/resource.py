from flask_restful import Resource
from padre.utils import DefaultLogger, ResourceDirectory
from padre.repository import PadreFileRepository
from flask import jsonify, Response
from authentication import requires_auth
from werkzeug.exceptions import HTTPException


repo = PadreFileRepository(ResourceDirectory().create_directory())

# Register a default logger
logger = DefaultLogger().get_default_logger()

class DatasetsAPI(Resource):  # Create a RESTful resource for datasets

    @requires_auth
    def get(self):  # Create GET endpoint
        logger.debug("start")
        logger.debug("end")
        return jsonify(data=repo.list())

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
