from functools import wraps
from flask import request, Response

from flask_dance.contrib.github import make_github_blueprint, github
from flask_dance.consumer import OAuth2ConsumerBlueprint


def check_basic_auth(username, password):
    """This function is called to check if a username /
    password combination is valid.
    """
    return username == 'admin' and password == 'secret'

def check_github_auth():
    """This function is called to check if a user /
    is logged in through github authentication.
    """
    if not github.authorized:
        #return redirect(url_for('github.login'))
        return False

    account_info = github.get('/user')
    if account_info.ok:
        return True
    return False

def check_orcid_auth():
    """This function is called to check if a user /
    is logged in through orcid authentication.
    """
    pass

def authenticate():
    """Sends a 401 response that enables basic auth"""
    return Response(
    'Could not verify your access level for that URL.\n'
    'You have to login with proper credentials', 401,
    {'WWW-Authenticate': 'Basic realm="Login Required"'})

def requires_auth(f):
    print('in requires_auth')
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_basic_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated

def get_github_blueprint():
    github_blueprint = make_github_blueprint(
        client_id="b0a17bb8355520a766d9",
        client_secret="1141371e055da2ee0cdfc39d43257610ddb6e9ee",
    )
    return github_blueprint

def get_orcid_blueprint():
    orcid_blueprint = OAuth2ConsumerBlueprint(
        "orcid", __name__,
        client_id="APP-6Y1RPQFTPK7T3DCA",
        client_secret="92388427-323a-4277-b952-ca28f5354911",
        base_url="https://pub.orcid.org/v2.0",
        token_url="https://orcid.org/oauth/token",
        authorization_url="https://orcid.org/oauth/authorize",
    )
    return orcid_blueprint

