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
import os
import tempfile
import uuid
from urllib.parse import urlparse

import arff
import networkx as nx
import openml as oml
import pandas as pd
from deprecated import deprecated
from google.protobuf.internal.encoder import _VarintBytes
from requests_toolbelt import MultipartEncoder

from pypadre.backend.http_experiments import HttpBackendExperiments
from pypadre.backend.serialiser import PickleSerializer
from pypadre.core.model.dataset.dataset import Dataset
from pypadre.core.model.dataset.attribute import Attribute
from pypadre.eventhandler import assert_condition, trigger_event
import pypadre.backend.protobuffer.protobuf.datasetV1_pb2 as proto


# TODO don't split backend for http if we don't split backend for file (stay consistent)
class PadreHTTPClient:

    def __init__(self, base_url="http://localhost:8080/api", user="", token=None
                 , online=False
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
        self._online = online
        if silent_codes is None:
            self.silent_codes = []
        else:
            self.silent_codes = silent_codes

        self._datasets_client = HTTPBackendDatasets(self)
        self._experiments_client = HttpBackendExperiments(self)

        self._access_token = token
        trigger_event('EVENT_WARN', condition=self._access_token is not None, source=self,
                      message="Authentication token is NONE. You need to authentication for user %s "
                              "with your current password (or set a new user)" % self.user)
        self._default_header['Authorization'] = self._access_token

    def authenticate(self, passwd="", user= None):
        """Authenticate user and retrieve token

        If current token is None or invalid, retrieve and set new token in request header

        :param passwd: Password for the user
        :type passwd: str
        :param user: User name if not provided then current user name will be used
        :type user: str
        :type passwd: str
        :returns: Token
        """
        if user is not None:
            self.user = user
        if self._access_token is None \
                or not self.is_token_valid(self._access_token):
            self._access_token = self.get_access_token(passwd)
        self._default_header['Authorization'] = self._access_token
        return self._access_token

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
                body["headers"]["Authorization"] = self._default_header["Authorization"]
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

    def get_base_url(self):
        url = self.base
        if url[-1] == "/":
            url = url[0:-1]
        return url

    def is_valid_url(self, url):
        """
        Validate if a url is valid.

        :param url: String containing url
        :returns: Boolean
        """
        result = False
        try:
            parsed = urlparse(url)
            if all([parsed.scheme, parsed.netloc]):
                result = True
        except:
            return result
        return result

    def get_access_token(self,  passwd=None):
        """Get access token.

        First get csrf token then use csrf to get oauth token.

        :param passwd: Password for the current user
        :type passwd: str
        :returns: Bearer token
        :rtype: str
        """
        token = None
        data = {
            "username": self.user,
            "password": passwd,
            "grant_type": "password",
            "scope": "read write"
        }
        parsed_base_url = urlparse(self.get_base_url())
        api = parsed_base_url.scheme + PadreHTTPClient.paths["padre-api"] + parsed_base_url.netloc
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
    def online(self):
        """
        sets the current online status of the client
        :return: True, if requests are passed to the server
        """
        return self._online

    @online.setter
    def online(self, online):
        self._online = online

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

    @deprecated("Use list instead")
    def list_datasets(self, search_name=None, search_metadata=None, start=0, count=999999999, search=None):
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

    @deprecated("Use put instead for protobuf support.")
    def put_dataset(self, dataset: Dataset, create :bool = True):
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

    def list(self, search_name=None, search_metadata=None, start=0, count=999999999, search=None) -> list:
        """
        List all data sets in the repository
        :param search_name: regular expression based search string for the title. Default None
        :param search_metadata: dict with regular expressions per metadata key. Default None
        """
        # todo apply the search metadata filter.
        data = []
        url = self.parent.get_base_url() + PadreHTTPClient.paths["search"]("datasets") + "name:?"
        if search_name is not None:
            url += search_name
        response = self.parent.do_get(url)
        content, links = self._parent.parse_hal(response)
        if "datasets" in content:
            for meta in content["datasets"]:

                binaries_url = self.parent.get_base_url() + PadreHTTPClient.paths["binaries"](str(meta["uid"]))
                pb_data = self.parent.do_get(binaries_url).content
                dataset = self.response_to_dataset(meta, pb_data)
                data.append(dataset)
        return data

    def put(self, dataset, create :bool = True):
        """Upload local dataset to the server and return dataset id
        '
        # Todo: Merge put and put dataset functions into one
        """
        data = dataset.metadata
        data["attributes"] = dataset.attributes
        url = self.parent.base[0:-1] + PadreHTTPClient.paths["datasets"]
        response = self.parent.do_post(url, **{"data":json.dumps(data)})
        dataset_id = response.headers["Location"].split("/")[-1]
        url = self.parent.base[0:-1] + PadreHTTPClient.paths["binaries"](dataset_id)
        with tempfile.TemporaryFile() as _file:
            binary = self.make_proto(dataset, _file)
            m = MultipartEncoder(
                fields={"field0": ("fname", binary, "application/x.padre.dataset.v1+protobuf")})
            response = self.parent.do_post(url, **{"data": m, "headers": {"Content-Type": m.content_type}})
        return dataset_id

    def get(self, _id, metadata_only=False):
        """Fetches data with given id from server and returns it"""
        url = self.parent.base[0:-1] + PadreHTTPClient.paths["dataset"](_id)
        response = self.parent.do_get(url)
        response_meta = json.loads(response.content.decode("utf-8"))
        binaries_url = self.parent.get_base_url() + PadreHTTPClient.paths["binaries"](_id)
        pb_data = self.parent.do_get(binaries_url).content
        dataset = self.response_to_dataset(response_meta, pb_data)
        trigger_event('EVENT_LOG', source=self, message="Loaded dataset " + _id + " from server:")
        return dataset

    def put_visualisation(self, id_, visualisation, description=None, supported_types=None):
        """
        Upload visualisation for given data set id.

        :param id_: Data set id for which visualisation should be uploaded
        :type id_: str
        :param visualisation: vega-lite specification
        :type visualisation: json
        :param description: Description of the visualisation
        :type description: str
        :param supported_types: Supported types
        :type supported_types: list
        :return: Http response or None
        """
        if description is None:
            description = "Visualization schema for dataset(%s)" % str(id_)
        if supported_types is None:
            supported_types = ["Multivariate data"]
        data = {"schema": visualisation,
                "description": description,
                "supportedTypes": supported_types}
        response = None
        dataset_visualization_url = self.parent.get_base_url() + PadreHTTPClient.paths["dataset-visualization"](str(id_))
        if self.parent.online:
            response = self.parent.do_post(dataset_visualization_url, **{"data": json.dumps(data)})
        return response

    def make_proto(self, dataset, _file):
        from pypadre.backend.protobuffer import proto_organizer
        pd_dataframe = dataset._binary.pandas_repr()
        pb_meta = proto.Meta()
        pb_meta.headers[:] = [str(header) for header in list(pd_dataframe)]
        pb_msg_serialized = pb_meta.SerializeToString()
        _file.write(_VarintBytes(len(pb_msg_serialized)))
        _file.write(pb_msg_serialized)
        _file.flush()
        for i, row in pd_dataframe.iterrows():
            pb_row = proto.DataRow()
            for i, entry in row.iteritems():
                proto_organizer.set_cell(pb_row, entry)
            serialize = pb_row.SerializeToString()

            _file.write(_VarintBytes(len(serialize)))
            _file.write(serialize)
            _file.flush()
        _file.seek(0)
        return _file

    def proto_to_dataframe(self, pb_data):
        from pypadre.backend.protobuffer import proto_organizer
        pb_pos = 0
        data_rows = []
        while pb_pos < len(pb_data):
            pb_row = proto.DataRow()
            pb_pos = proto_organizer.read_delimited_pb_msg(pb_data, pb_pos, pb_row)
            data_fields = []
            for cell in pb_row.cells:
                field = cell.WhichOneof("cell_type")
                if field is None:
                    data_fields.append(field)
                else:
                    value = getattr(cell, cell.WhichOneof("cell_type"))
                    data_fields.append(value)
            data_rows.append(data_fields)
        df = pd.DataFrame(data_rows)
        return df

    def response_to_dataset(self, meta, binary):
        from pypadre import graph_import
        attribute_name_list = []
        atts = []
        meta_attributes = meta["attributes"]
        if meta_attributes[0]["defaultTargetAttribute"]:
            meta_attributes = list(reversed(meta_attributes))
        for attr in meta_attributes:
            atts.append(Attribute(**attr))
            attribute_name_list.append(attr["name"])

        df_data = self.proto_to_dataframe(binary)
        del meta["attributes"]
        dataset = Dataset(meta["uid"], **meta)
        if df_data.empty:
            return dataset
        df_data.columns = attribute_name_list

        if dataset.isgraph:
            node_attr = []
            edge_attr = []
            for attr in atts:
                graph_role = attr.context["graph_role"]
                if graph_role == "source":
                    source = attr.name
                elif graph_role == "target":
                    target = attr.name
                elif graph_role == "nodeattribute":
                    node_attr.append(attr.name)
                elif graph_role == "edgeattribute":
                    edge_attr.append(attr.name)
            network = nx.Graph() if meta["type"] == "graph" else nx.DiGraph()
            graph_import.pandas_to_networkx(df_data, source, target, network, node_attr, edge_attr)
            dataset.set_data(network, atts)
        else:
            dataset.set_data(df_data, atts)
        return dataset

    def load_oml_dataset(self, did):
        """Load dataset from openML with given id.

        :param did: Dataset Id on openML
        :type did: str
        :return: Padre compatible dataset
        :rtype: <class 'pypadre.datasets.Dataset'>
        """
        from pypadre.app.padre_app import p_app
        path = os.path.expanduser(p_app.config.get("root_dir", "LOCAL BACKEND")) + '/temp/openml'
        oml.config.apikey = p_app.config.get("oml_key", "GENERAL")
        oml.config.cache_directory = path
        dataset = None
        try:
            load = oml.datasets.get_dataset(did)
            meta = dict()
            meta["id"] = str(uuid.uuid4())
            meta["name"] = load.name
            meta["version"] = load.version
            meta["description"] = load.description
            meta["originalSource"] = load.url
            meta["type"] = "Multivariat"
            meta["published"] = True
            dataset = Dataset(meta["id"], **meta)
            raw_data = arff.load(open(load.data_file, encoding='utf-8'))
            attribute_list = [att[0] for att in raw_data["attributes"]]
            df_data = pd.DataFrame(data=raw_data['data'])
            df_data.columns = attribute_list
            target_features = load.default_target_attribute.split(",")
            for col_name in target_features:
                df_data[col_name] = df_data[col_name].astype('category')
                df_data[col_name] = df_data[col_name].cat.codes

            atts = []
            for feature in df_data.columns.values:
                atts.append(Attribute(name=feature,
                                      measurementLevel="Ratio" if feature in target_features else None,
                                      defaultTargetAttribute=feature in target_features))
            dataset.set_data(df_data, atts)

        except ConnectionError as err:
            assert_condition(condition=False, source=self, message="openML unreachable! \nErrormessage: " + str(err))

        return dataset


PadreHTTPClient.paths = {
    "padre-api": "://padre-api:@",
    "datasets": "/datasets",
    "experiments": "/experiments",
    "experiment": lambda id: "/experiments/" + id + "/",
    "projects": "/projects",
    "results": lambda e_id, r_id, rs_id: "/experiments/" + e_id + "/runs/" + r_id + "/splits/" + rs_id + "/results",
    "results-json": lambda e_id, r_id, rs_id: "/experiments/" + e_id + "/runs/" + r_id + "/splits/" + rs_id + "/results/json",
    "experiment-runs": lambda e_id: "/experiments/" + e_id + "/runs/",
    "runs": "/runs",
    "run": lambda e_id, r_id: "/experiments/" + e_id + "/runs/" + r_id,
    "run-models": lambda e_id, r_id: "/experiments/" + e_id + "/runs/" + r_id + "/model",
    "experiment-run-splits": lambda e_id, r_id: "/experiments/" + e_id + "/runs/" + r_id + "/splits",
    "run-splits": "/runSplits",
    "run-split": lambda e_id, r_id, rs_id: "/experiments/" + e_id + "/runs/" + r_id + "/splits/" + rs_id,
    "search": lambda entity: "/" + entity + "/search?search=",
    "oauth-token": lambda csrf_token: "/oauth/token?=" + csrf_token,
    "splits": "/splits",
    "split": lambda id: "/splits/" + id,
    "dataset": lambda id: "/datasets/" + id + "/",
    "dataset-visualization": lambda did: "/datasets/" + did + "/visualizations",
    "binaries": lambda id: "/datasets/" + id + '/binaries/',
    "visualizations": "/visualizations"
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


