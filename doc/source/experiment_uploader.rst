=================
Upload Experiment
=================

.. automodule:: backend.experiment_uploader


.. autoclass:: backend.experiment_uploader.ExperimentUploader
   :members:

Upload Experiment To Server
--------------------------
This module is used with http backend. Purpose of this module is to upload or download experiment
from server.

To upload experiment to server, project and dataset should also be created on the server for
the current experiment which is being uploaded.

Following functions are implemented

#. create_project: Create project on the server
#. create_dataset: Create dataset on the server
#. put_experiment: Upload experiment to the server

Todo: Delete, Get and List experiments from server