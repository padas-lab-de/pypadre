=================
Config
=================

.. automodule:: app.padre_app


.. autoclass:: app.padre_app.PadreConfig
   :members:

Padre Config
-------------------
PadreConfig class covering functionality for viewing or updating default
configurations for PadreApp.
Configuration file is placed at ~/.padre.cfg

Expected values in config are following


[HTTP]

user = username

passwd = user_password

base_url = http://localhost:8080/api

token = oauth_token

[FILE_CACHE]

root_dir = ~/.pypadre/



Implemented functionality
-------------------------

#. Get list of dicts containing key, value pairs for all sections in config
#. Get value for given key.
#. Set value for given key in config
#. Authenticate given user and update new token in the config


Using config through CLI
------------------------------------

#. Use **get_config_param** command to get value of given param

   * param: name of attribute
#. Use **set_config_param** command to set value of given param

   * param: must me be a tuple of key value pair
#. Use **get_config_params** command to get list of all params
#. Use **authenticate** to set new token in config. If no params are provided default settings will be used

   * url: Url of server api default None
   * user: User name on server default None
   * passwd: Password for given user default None
