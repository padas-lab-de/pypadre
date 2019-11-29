=================
Config
=================

.. automodule:: pypadre.pod.app.core_app


.. autoclass:: pypadre.pod.app.config.padre_config.PadreConfig
   :members:

Padre Config
-------------------
PadreConfig class covering functionality for viewing or updating default
configurations for PadreApp.
Configuration file is placed at ~/.padre.cfg

When running the tests ~/.padre-test.cfg is created which is deleted when the tests are completed.

Expected values in config are following

[LOCAL BACKEND]

root_dir = ~/.pypadre/

[GENERAL]

offline = True

oml_key = openml_key_here



Implemented functionality
-------------------------

#. Get all sections from config
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
