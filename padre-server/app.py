"""
Web application main file.
This file contains registration of resource, mapping of resource and main method.
"""
# TODO this is just template code and a proof-of-concept
from flask import Flask, jsonify, redirect, url_for
from flask_restful import Api
from flask_dance.contrib.github import github
from werkzeug.exceptions import default_exceptions
from werkzeug.exceptions import HTTPException

from db_service import Setup
from padre.utils import DefaultLogger
from padre.constants import DATASET_RESOURCE_MAPPING, DATASETS_RESOURCE_MAPPING, DEBUG
from resource import DatasetsAPI, DatasetAPI
from authentication import get_github_blueprint, get_orcid_blueprint


debug = True
app = Flask(__name__)  # Create a Flask WSGI appliction
api = Api(app)  # Create a Flask-RESTPlus API

app.secret_key = "supersekrit"

app.register_blueprint(get_orcid_blueprint())

app.register_blueprint(get_github_blueprint(), url_prefix="/login")

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


# Below OAuth authentication section to be moved to client code. This is a sample usage of the functionality.
# We have a decorator implemented in authentication.py file.

@app.route('/home')
def github_home():
    return "home"

#'https://orcid.org/login'

@app.route('/orcid')
def orcid_login():
    logger.debug('orcid_login : START')
    if not get_orcid_blueprint().session.authorized:
        #return "unauthorized orcid attempt"
        logger.debug('in orcid_login before redirect')
        #'https://orcid.org/oauth/authorize?client_id=APP-6Y1RPQFTPK7T3DCA&response_type=code&scope=/authenticate&redirect_uri=http://localhost:5000/orcid/authorized'
        logger.debug('orcid_login : END')
        return redirect(url_for('orcid.login'))
    logger.debug('authorized')
    account_info = get_orcid_blueprint().session.get('/user')
    if account_info.ok:
        account_info_json = account_info.json()
        logger.debug('orcid_login : END')
        return '<h1>Your orcid info is {}'.format(account_info_json)
    logger.debug('orcid_login : END')
    return '<h1>Request failed!</h1>'


@app.route('/github1')
def github_login():
    logger.debug('github_login : START')
    if not github.authorized:
        logger.debug('unauthorized : redirecting to github login')
        logger.debug('github_login : END')
        return redirect(url_for('github.login'))

    account_info = github.get('/user')
    if account_info.ok:
        account_info_json = account_info.json()
        logger.debug('authorized')
        logger.debug('github_login : END')
        return '<h1>Your Github name is {}'.format(account_info_json['login'])
    logger.debug('github_login : END')
    return '<h1>Request failed!</h1>'

# OAuth authentication section end.

# Register all Resources and their mapping here
api.add_resource(DatasetsAPI, DATASETS_RESOURCE_MAPPING, endpoint="datasets")
api.add_resource(DatasetAPI, DATASET_RESOURCE_MAPPING, endpoint="dataset")

# Shutdown hook
# atexit.register(Setup.close_setup())

# App main function
if __name__ == '__main__':
    handle_exceptions()
    Setup.init_setup()
    app.run(debug=DEBUG)  # Start a development padre-server