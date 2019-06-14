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


[HTTP BACKEND]

user = username

base_url = http://localhost:8080/api

token = oauth_token

[LOCAL BACKEND]

root_dir = ~/.pypadre/

[GENERAL]

offline = True

oml_key = openml_key_here



Implemented functionality
-------------------------

#. Get all sections from config
#. Get value for given key for given section(default HTTP BACKEND).
#. Set value for key in config for given section(default HTTP BACKEND)
#. Save config


Using config through CLI
------------------------------------

#. Use **get_config_param** command to get value of given param

   * param: name of attribute
   * section: name of the section, if its None then default section will be used

#. Use **set_config_param** command to set value of given param

   * param: must be a tuple of key value pair
   * section: name of the section, if its None then default section will be used

#. Use **list_config_params** command to get list of all params for given section

    * section: name of the section
