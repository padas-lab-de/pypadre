"""
Web application main file.
This file contains registration of resource, mapping of resource and main method.
"""
# TODO this is just template code and a proof-of-concept
from flask import Flask, jsonify
from flask_restful import Api

from padre.utils import DefaultLogger
from padre.constants import DATASET_RESOURCE_MAPPING, DATASETS_RESOURCE_MAPPING, DEBUG
from resource import DatasetsAPI, DatasetAPI
from werkzeug import exceptions as wexceptions
from werkzeug.exceptions import default_exceptions
from werkzeug.exceptions import HTTPException

debug = True
app = Flask(__name__)  # Create a Flask WSGI appliction
api = Api(app)  # Create a Flask-RESTPlus API

# Register a default logger
logger = DefaultLogger().get_default_logger()

# Ensure all error responses are JSON
def handle_exceptions():
    """
        Creates a JSON-oriented Flask app.

        All error responses that you don't specifically
        manage yourself will have application/json content
        type, and will contain JSON like this (just an example):

        { "message": "405: Method Not Allowed" }
    """
    def _json_error(ex):
        code = ex.code if isinstance(ex, HTTPException) else 500
        response = jsonify(message=str(ex))
        response.status_code(code)
        return response

    for code in default_exceptions.keys():
        app.error_handler_spec[None][code] = _json_error

# Register all Resources and their mapping here

api.add_resource(DatasetsAPI, DATASETS_RESOURCE_MAPPING, endpoint="datasets")
api.add_resource(DatasetAPI, DATASET_RESOURCE_MAPPING, endpoint="dataset")

# App main function

if __name__ == '__main__':
    handle_exceptions()
    app.run(debug=DEBUG)  # Start a development padre-server
