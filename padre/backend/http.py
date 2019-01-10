"""
backend for http padre requests.

Responsible point of interaction to the Padre RESTFull API

Manages:
- Configuration
- Basic Requests
- Object validation
- Object conversion (via factory)
"""
# todo: add logging
# todo url management is not perfect yet.
import requests as req
import json
import io

from deprecated import deprecated
from padre.backend.http_experiments import HttpBackendExperiments
from padre.backend.serialiser import PickleSerializer
from padre.datasets import Dataset, Attribute
import logging

logger = logging.getLogger('pypadre - http')

class PadreHTTPClient:

    def __init__(self, base_url="http://localhost:8080/api", user="", token=None
                 , silent_codes=None
                 , default_header={'content-type': 'application/hal+json'}):
        if base_url.endswith("/"):
            self.base = base_url
        else:
            self.base = base_url + "/"
        self.user = user
        self.last_status = None
        self._data_serializer = PickleSerializer
        self._default_header = default_header
        self._access_token = None
        if silent_codes is None:
            self.silent_codes = []
        else:
            self.silent_codes = silent_codes

        self._datasets_client = HTTPBackendDatasets(self)
        self._experiments_client = HttpBackendExperiments(self)

        self._access_token = token
        if self._access_token is not None:
            self._default_header['Authorization'] = self._access_token
        else:
            logger.warning("Authentication token is NONE. You need to authentication for user %s "
                           "with your current password (or set a new user)" % self.user)

    def authenticate(self, passwd="", user= None):
        if user is not None:
            self.user = user
        if self._access_token is None \
                or not self.is_token_valid(self._access_token):
            self._access_token = self.get_access_token(passwd)
        self._default_header['Authorization'] = self._access_token

    def do_request(self, request, url, **body):
        """
        Do a request.
        :param request: request function (get, post, etc.)
        :param url: url of the request. if url does start with http it is considered as absolut, otherwise it is considered as relativ to the api endpoint
        :param status_code_silent: status codes that DONT raise an exceptoin
        :param body: valid body for the request.
        """
        if not url.startswith("http"):
            url = self.join_url(url)
        if body is None:
            r = request(url)
        else:
            if "headers" not in body:
                body["headers"] = self._default_header
            if "Authorization" not in body["headers"]:
                body["headers"]["Authorization"] = self._access_token
            r = request(url, **body)
        self.last_status = r.status_code
        r.raise_for_status()
        return r

    def do_get(self, url, **body):
        return self.do_request(req.get, url, **body)

    def do_put(self, url, **body):
        return self.do_request(req.put, url, **body)

    def do_post(self, url, **body):
        return self.do_request(req.post, url, **body)

    def do_patch(self, url, **body):
        return self.do_request(req.patch, url, **body)

    def do_delete(self, url, **body):
        return self.do_request(req.delete, url, **body)

    """
    returns the paging part of a url beginning with '?'
    """
    def get_paging_url(self, page=None, size=None, sort=None):
        ret = ""
        sep = "?"
        if page is not None:
            ret = ret + sep + "page=" + page
            sep = ","
        if size is not None:
            ret = ret + sep + "size=" + size
            sep = ","
        if sort is not None:
            ret = ret + sep + "sort=" + sort
        return ret

    def join_url(self, *urls):
        return self.base + "/".join(urls)

    """
    parses the hal format of the request response. Depending on the response type, the item itself or a set of items is returned
    :param result: valid json formatted result of the request
    :return: (content, links) tuple, where content is either an object or a list of objects and links is a dictionary of links.
    """

    def parse_hal(self, result):
        j = result.json()
        if "_embedded" in j:
            return j["_embedded"], j["_links"]
        else:
            links = j["_links"]
            del j["_links"]
            return j, links

    def get_dataset(self, datasetid, download=True, format="numpy"):
        """
        get the dataset with the specified id
        :param id: either a number as id of the dataset or a url pointing to the resource
        :param download: flag indicating whether to download all records / binary
        :param format: format of the records / binary
        :return: Dataset
        """
        res = self.do_get(self._get_id_url(datasetid, "dataset"))
        content, links = self.parse_hal(res)
        dataset = json2dataset(content, links)
        if download:
            self.download_binary(dataset, format)
        return dataset

    def get_dataset_formats(self, datasetid):
        """
        :param id: id of the dataset either as absolute url or as number id
        :return: list of strings
        """
        res = self.do_get(self._get_id_url(datasetid, "binaries", "binaries/"))
        content, links = self.parse_hal(res)
        if "content" in content:
            return [d["format"] for d in content["content"]]
        else:
            return []

    def download_binary(self, dataset, format):
        res = self.do_get(self._get_id_url(dataset.id, "binaries", "binaries/") + format,
                          headers={"content-type": "application/octet-stream"})
        dataset.set_data(self._data_serializer.deserialize(res.content),
                         dataset.attributes)

    @deprecated(reason ="use datasets.put method")
    def upload_dataset(self, dataset, create=False):
        self.datasets.put(dataset, create)

    def _get_id_url(self, id, kind, postfix=""):
        if str(id).startswith("http"):
            if not id.endswith("/"):
                id = id + "/"
            return id + postfix
        else:
            return PadreHTTPClient.paths[kind](id)

    def get_access_token(self,  passwd=None):
        """Get access token.

        First get csrf token then use csrf to get oauth token.

        :returns: Bearer token
        :rtype: str
        """
        token = None
        data = {
            "username": self.user,
            "password": passwd,
            "grant_type": "password"
        }
        api = PadreHTTPClient.paths["padre-api"]
        try:
            csrf_token = self.do_get(api).cookies.get("XSRF-TOKEN")
            url = api + PadreHTTPClient.paths["oauth-token"](csrf_token)
            response = self.do_post(url,
                                    **{'data': data,
                                       'headers': {'content-type': 'application/x-www-form-urlencoded'}
                                       })
        except req.exceptions.RequestException as e:
            print(str(e))  # todo: Handle failed calls properly
            return token

        if response.status_code == 200:
            token = "Bearer " + json.loads(response.content)['access_token']
        return token

    def has_token(self):
        if self._access_token is not None:
            return True
        return False

    def is_token_valid(self, token):
        """
        Check if given token is valid
        :param token:
        :return:
        """
        result = False
        if token is None:
            return result
        try:
            response = self.do_get(self.base + "users/me", **{"headers": {"Authorization": token}})
        except req.exceptions.HTTPError as e:
            return result
        if response.status_code == 200:
            result = True
        return result

    @property
    def experiments(self):
        return self._experiments_client

    @property
    def datasets(self):
        return self._datasets_client


class HTTPBackendDatasets:

    def __init__(self, parent):
        self._parent = parent

    @property
    def parent(self):
        return self._parent

    def list(self, search_name=None, search_metadata=None, start=0, count=999999999, search=None):
        """
        lists all datasets available remote giving the search parameters.
        :param search_name:
        :param search_metadata:
        :param start:
        :param count:
        :param search:
        :return:
        """
        ret = self._parent.do_get(PadreHTTPClient.paths["datasets"])
        content, links = self._parent.parse_hal(ret)
        if "datasets" in content:
            return [json2dataset(ds) for ds in content["datasets"]]
        else:
            return []

    def put(self, dataset: Dataset, create :bool = True):
        assert create or dataset.id is not None  # dataset id must be provided when the dataset should be updated
        payload = dict()
        payload.update(dataset.metadata)
        payload["attributes"] = []
        for ix, a in enumerate(dataset.attributes):
            a_json = dict()
            a_json["name"] = a.name
            a_json["index"] = ix
            a_json["measurementLevel"] = a.measurementLevel
            a_json["unit"] = a.unit
            a_json["description"] = a.description
            a_json["defaultTargetAttribute"] = a.defaultTargetAttribute
            payload["attributes"].append(a_json)
        if create:
            res = self._parent.do_post(PadreHTTPClient.paths["datasets"], data=json.dumps(payload))
        else:
            if str(dataset.id).startswith(("http")):
                res = self._parent.do_put(dataset.id, data=json.dumps(payload))
            else:
                res = self._parent.do_put(PadreHTTPClient.paths["dataset"](dataset.id), data=json.dumps(payload))
        dataset.id(self._parent.parse_hal(res)[1]["self"]["href"])
        if dataset.has_data():
            content, links = self._parent.parse_hal(res)
            # todo check format, compare it with the returned binary links possible, then submit.
            if not "binaries" in links:
                link = links["self"]["href"] + "/binaries/" + dataset.binary_format() + "/"
            else:
                link = links["binaries"]["href"] + "/" + dataset.binary_format() + "/"
            self._parent.do_put(link,
                        headers={},  # let request handle the content type
                        files={"file": io.BytesIO(self._parent._data_serializer.serialise(dataset.data))})

PadreHTTPClient.paths = {
    "padre-api": "http://padre-api:@localhost:8080",
    "datasets": "/datasets",
    "experiments": "/experiments",
    "experiment": lambda id: "/experiments/" + id + "/",
    "projects": "/projects",
    "results": lambda e_id, r_id, rs_id: "/experiments/" + e_id + "/runs/" + r_id + "/splits/" + rs_id + "/results",
    "runs": "/runs",
    "run-models": lambda e_id, r_id: "/experiments/" + e_id + "/runs/" + r_id + "/model",
    "run-splits": "/runSplits",
    "oauth-token": lambda csrf_token: "/oauth/token?=" + csrf_token,
    "splits": "/splits",
    "dataset": lambda id: "/datasets/" + id + "/",
    "binaries": lambda id: "/datasets/" + id + '/binaries/',
}


def json2dataset(json, links=None):
    """
    creates a dataset object and parses the json dict
    :param json: dict
    :param links: Links from HAL description with self link (for id)
    :return:
    """
    if links is None and "_links" in json:
        links = json["_links"]
    if links is not None and "self" in links:
        _id = links["self"]["href"]
    else:
        _id = None
    attributes = json["attributes"]
    ds = Dataset(_id, **dict((k, json[k]) for k in json.keys()
                             if k not in ("datasetId", "attributes", "binaries", "experiments", "_links")))
    sorted(attributes, key=lambda a: a["index"])
    assert sum([int(a["index"]) for a in attributes]) == len(attributes) * (
    len(attributes) - 1) / 2  # check attribute correctness here
    ds.set_data(None,
                [Attribute(a["name"], a["measurementLevel"], a["unit"], a["description"], a["defaultTargetAttribute"])
                 for a in attributes])
    return ds
