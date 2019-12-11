import ast
import configparser
import os

from pypadre.pod.util.dict_util import dict_merge

if "PADRE_BASE_URL" in os.environ:
    _BASE_URL = os.environ["PADRE_BASE_URL"]
else:
    _BASE_URL = "http://localhost:8080/api"

if "PADRE_CFG_FILE" in os.environ:
    _PADRE_CFG_FILE = os.environ["PADRE_CFG_FILE"]
else:
    _PADRE_CFG_FILE = os.path.expanduser('~/.padre.cfg')

_GITLAB_BASE_URL = 'http://localhost:8080/'


class PadreConfig:
    """
    PadreConfig class covering functionality for viewing or updating default
    configurations for PadreApp.
    Configuration file is placed at ~/.padre.cfg

    Expected values in config are following
    ---------------------------------------
    [HTTP BACKEND]\n
    user = username\n
    passwd = user_password\n
    base_url = http://localhost:8080/api\n
    token = oauth_token

    [LOCAL BACKEND]\n
    root_dir = ~/.pypadre/

    [GITLAB BACKEND]\n
    root_dir = ~/.pypadre/\n
    user = username\n
    token = user_private_token\n
    gitlab_url = https://your_gitlab_server


    [GENERAL]
    offline = True
    oml_key = openML_api_key
    ---------------------------------------

    Implemented functionality.

    1- Get list of dicts containing key, value pairs for all sections in config
    2- Get value for given key.
    3- Set value for given key in config
    4- Authenticate given user and update new token in the config
    """

    def __init__(self, config_file: str = _PADRE_CFG_FILE, create: bool = True, config: dict = None):
        """
        PRecedence of Configurations: default gets overwritten by file which gets overwritten by config parameter
        :param config: str pointing to the config file or None if no config file should be used
        :param create: true if the config file should be created
        :param config: Additional configuration
        """
        self._config = self.default()

        # handle file here
        self._config_file = config_file
        if self._config_file is not None:
            self.__load_config()
            if not os.path.exists(self._config_file) and create:
                self.save()

        # now merge
        self.__merge_config(config)
        self.save()

    def __merge_config(self, to_merge):
        # merges the provided dictionary into the config.
        if to_merge is not None:
            dict_merge(self._config, to_merge)

    def __load_config(self):
        """
        loads a padre configuration from the given file or from the standard file ~/.padre.cfg if no file is provided
        :param config_file: filename of config file
        :return: config accessable as dictionary
        """
        config = configparser.ConfigParser()
        if os.path.exists(self._config_file):
            config.read(self._config_file)
            config_data = dict(config._sections)
            if config.has_option("GENERAL", "offline"):
                config_data["GENERAL"]["offline"] = config.getboolean("GENERAL", "offline")
            self.__merge_config(config_data)

    def default(self):
        """
        :return: default values of the config
        """
        return {
            "GENERAL": {
                "offline": True,
                "backends": [
                    {
                        "root_dir": os.path.join(os.path.expanduser("~"), ".pypadre")
                    }
                ]
            }
        }

    @property
    def config_file(self):
        return self._config_file

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config):
        self._config = config

    @property
    def general(self):
        return self._config["GENERAL"]

    def load(self) -> None:
        """
        reloads the configuration specified under the current config path
        """
        self.__load_config()

    def save(self) -> None:
        """
        saves the current configuration to the configured file
        """
        pconfig = configparser.ConfigParser()
        for k, v in self._config.items():
            pconfig[k] = v
        with open(self._config_file, "w") as cfile:
            pconfig.write(cfile)

    def sections(self) -> list:
        """
        :return: returns all sections of the config file
        """
        return self._config.keys()

    def set(self, key, value, section='GENERAL'):
        """
        Set value for given key in config

        :param key: Any key in config
        :type key: str
        :param value: Value to be set for given key
        :type value: str
        :param section: Section to be changed in config, default HTTP
        :type section: str
        """
        if not self.config[section]:
            self.config[section] = dict()
        self.config[section][key] = value

    def has_entry(self, key, section='GENERAL'):
        try:
            self.get(key, section)
            return True
        except:
            return False

    def get(self, key, section='GENERAL'):
        """
        Get value for given key.
        :param section: Section to be queried
        :param key: Any key in config for any section
        :type key: str
        :return: Found value or False
        """
        # TODO lists seem not be be supported in base config files of python. Don't hardcode
        if key == "backends":
            return ast.literal_eval(self._config[section][key])
        return self._config[section][key]

    def get_list(self, key, section='GENERAL') -> list:
        """
        Get value for given key.
        :param section: Section to be queried
        :param key: Any key in config for any section
        :type key: str
        :return: Found value or False
        """
        return self._config[section][key]
