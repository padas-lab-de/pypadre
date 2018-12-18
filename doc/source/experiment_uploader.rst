=================
Upload Experiment
=================

..automodule:: backend.experiment_uploader


..autoclass:: backend.experiment_uploader.ExperimentUploader
   :members:

Upload Experiment To Server
----------------------------
This module is used with http backend. Purpose of this module is to upload or download experiment, run, run-splits and results
to server.

To upload experiment to server, project and dataset should also be created on the server for
the current experiment which is being uploaded.

Following main functions are implemented

#. get_or_create_project: Get or create project on the server if not exists
#. get_or_create_dataset: Get or create dataset on the server if not exists
#. put_experiment: Upload experiment including hyperparameters to the server and return url of the experiment
#. put_run: Upload run including hyperparameters and run model(as binary) to the server and return url of the run
#. put_split: Upload split information to the server and return url of the split
#. put_results: Convert results to protobuf and upload them to the server for each execution of run-split

Todo: Delete, Get and List experiments from server