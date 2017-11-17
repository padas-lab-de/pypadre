from flask_restful import Resource
from padre.utils import DefaultLogger, ResourceDirectory
from padre.repository import PadreFileRepository
from padre.ds_import import load_sklearn_toys
from padre.datasets import new_dataset
from padre.constants import DEBUG
from flask import jsonify, request, Response, abort
from authentication import requires_auth


repo = PadreFileRepository(ResourceDirectory().create_directory())

if DEBUG:
     for i in load_sklearn_toys():
         repo.put(i.name, i)

# Register a default logger
logger = DefaultLogger().get_default_logger()

class DatasetsAPI(Resource):  # Create a RESTful resource for datasets

    @requires_auth
    def get(self):  # Create GET endpoint
        logger.debug("start")
        logger.debug("end")
        # TODO: Need to call the json_response to form a complete response object
        return jsonify(data=repo.list())

    def put(self):  # update a new dataset
        pass

    def post(self):  # update a new dataset
        files = request.files  # retrieves the file attachment from request body
        name = request.form['name']  # retrieves the string parameter from request body
        # trying to mock the attributes.
        metadata = {"format": "numpy", "data": files['param2']}
        dataset = new_dataset(name, metadata, files['param1'], files['param3'])
        repo.put(name, dataset)
        # TODO: Need to call the json_response to form a complete response object
        return jsonify(data=repo.list())

    def delete(self):  # delete a new dataset
        pass


class DatasetAPI(Resource):  # Create a RESTful resource for datasets

    def get(self, name):  # Create GET endpoint
        print(name)
        return repo.get(name).metadata

    def put(self):  # update a new dataset
        data = request.json
        print(data)
        pass

    def post(self):  # update a new dataset
        data = request.json
        print(data)
        pass

    def delete(self):  # delete a new dataset
        pass
