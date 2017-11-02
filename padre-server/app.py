"""
Web application main file.
"""
# TODO this is just template code and a proof-of-concept
from flask import Flask
from flask_restful import Resource, Api
import os
from padre.repository import PadreFileRepository

# DEBUG STUFF
from padre.ds_import import load_sklearn_toys

debug = True
app = Flask(__name__)                  #  Create a Flask WSGI appliction
api = Api(app)                         #  Create a Flask-RESTPlus API

# TODO create a corresponding configuration object. look up best practices
data_dir = os.path.expanduser("~/tmp/padre_srv/data")
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

app.repo = PadreFileRepository(data_dir)
if debug:
    for i in load_sklearn_toys():
        app.repo.put(i.name, i)

class DatasetsAPI(Resource):            #  Create a RESTful resource for datasets

    def get(self):                         #  Create GET endpoint
        return  app.repo.list()

    def put(self):                         # update a new dataset
        pass

    def post(self):  # update a new dataset
        pass

    def delete(self):  # delete a new dataset
        pass

class DatasetAPI(Resource):  # Create a RESTful resource for datasets

    def get(self, name):  # Create GET endpoint
        print(name)
        return app.repo.get(name).metadata

    def put(self):  # update a new dataset
        pass

    def post(self):  # update a new dataset
        pass

    def delete(self):  # delete a new dataset
        pass



api.add_resource(DatasetsAPI, "/api/datasets/", endpoint="datasets")
api.add_resource(DatasetAPI, "/api/datasets/<name>", endpoint="dataset")

if __name__ == '__main__':
    app.run(debug=debug)                #  Start a development padre-server
