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