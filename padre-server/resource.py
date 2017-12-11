from flask_restful import Resource
from padre.utils import DefaultLogger
from padre.datasets import new_dataset
from padre.constants import DEBUG
from flask import jsonify, request, Response, abort
from authentication import requires_auth
from db_service import DataSetService
from padre.repository import PickleSerializer, JSonSerializer
from padre.utils import DefaultLogger, ResourceDirectory
import os

# Register a default logger
logger = DefaultLogger().get_default_logger()

class DatasetsAPI(Resource):  # Create a RESTful resource for datasets

    @requires_auth
    def get(self):  # Create GET endpoint
        logger.debug("start")
        logger.debug("end")
        # TODO: Need to call the json_response to form a complete response object
        return jsonify(data=DataSetService().getAll())

    def put(self):  # update a new dataset
        pass

    def post(self):  # update a new dataset
        try:
            # temp_dir = ResourceDirectory().create_directory()
            dataset_name = request.form["name"]
            for upload in request.files.getlist("file"):

                print("{} is the file name".format(upload.filename))
                filename = upload.filename

                # This is to verify files are supported
                ext = os.path.splitext(filename)[1]
                if (ext == ".json") or (ext == ".bin"):
                    print("File supported, moving on...")
                    if (ext == ".json"):
                        metadata = JSonSerializer.deserialize(upload.read())
                    if (ext == ".bin") and (filename == "data.bin"):
                        data = PickleSerializer.deserialize(upload.read())
                    if (ext == ".bin") and (filename == "target.bin"):
                        target = PickleSerializer.deserialize(upload.read())
                else:
                    raise TypeError("Unsupported File Type " + ext)
                print("Accept incoming file:", filename)
        except Exception as e:
            print(e)

        # some preprocessing if needed to check if uploaded dataset is a valid 'Numpy dataset'.
        dataset = new_dataset(dataset_name, metadata, data, target)
        DataSetService().save(dataset_name, dataset)
        # TODO: Need to call the json_response to form a complete response object
        return jsonify(data=DataSetService().getAll())

    def delete(self):  # delete a new dataset
        pass


class DatasetAPI(Resource):  # Create a RESTful resource for datasets

    def get(self, name):  # Create GET endpoint
        print(name)
        return jsonify(data=DataSetService().get(name))

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
