import requests
from constants import BASE_URL, DATASETS, DATASET, BASIC_AUTH
from padre.utils import DefaultLogger, ResourceDirectory

temp_path = ResourceDirectory().create_directory()


class RestClient(object):

    def list_datasets(self, resource_group_name=DATASETS):
        response = get_resources(resource_group_name)
        print('Response status : ', response[1])
        print('Response Headers : ', response[2])
        print('Response enconding : ', response[3])

    def fetch_dataset(self, name):
        response = get_resource(name)
        print('Response status : ', response[1])
        print('Response Headers : ', response[2])
        print('Response enconding : ', response[3])

    def post(self, dataset_name, dataset_data_path, dataset_metadata_path, dataset_target_path):
        response = post_req(dataset_name, dataset_data_path, dataset_metadata_path, dataset_target_path)
        print('Response status : ', response[1])
        print('Response Headers : ', response[2])
        print('Response enconding : ', response[3])

    def put(self, dataset):
        pass


    def delete(self, name):
        pass


def get_resources(resource_group_name):
    response = requests.get(BASE_URL+resource_group_name, auth=BASIC_AUTH)
    data = response.json()
    return data, response.status_code, response.headers['content-type'], response.encoding


def get_resource(resource_name):
    response = requests.get(BASE_URL+DATASET+resource_name, auth=BASIC_AUTH)
    data = response.json()
    return data, response.status_code, response.headers['content-type'], response.encoding


def post_req(name="defaultTest",
            dataset_data_path=temp_path + "/Boston House Prices dataset/data.bin",
            dataset_metadata_path=temp_path + "/Boston House Prices dataset/metadata.json",
            dataset_target_path=temp_path + "/Boston House Prices dataset/target.bin"):

    data = {"name": name}
    files = {('file', open(dataset_data_path, 'rb')),
             ('file', open(dataset_metadata_path, 'rb')),
             ('file', open(dataset_target_path, 'rb'))}

    response = requests.post(BASE_URL + DATASETS, files=files, data=data, auth=BASIC_AUTH)
    # Get the response as json list
    # TODO: check if response is 200 and then fetch the json.
    data = response.json()
    return data, response.status_code, response.headers['content-type'], response.encoding
    # print('Response status : ', response.status_code)
    # print('Response Headers : ', response.headers['content-type'])
    # print('Response enconding : ', response.encoding)


# flow = OAuth2WebServerFlow(client_id="APP-6Y1RPQFTPK7T3DCA",
#                            client_secret="92388427-323a-4277-b952-ca28f5354911",
#                            scope=None,
#                            redirect_uri='http://localhost:5000/login/github/authorized')
# auth_uri = flow.step1_get_authorize_url()
# # Redirect the user to auth_uri on your platform.
#
# credentials = flow.step2_exchange(code)

if __name__ == "__main__":
    # RestClient().fetch_dataset("Linnerrud dataset")
    # get_resources()
    # get_resource()
    post_req()
