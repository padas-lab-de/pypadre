Experiment Storage and Online Synchronization
=============================================


Local Storage Structure
-----------------------


Server Synchronisation
----------------------

Upload local experiment
***********************
Local experiment can be uploaded to the server if its not already uploaded.
In the app :func:`pypadre.app.padre_app.ExperimentApp.upload_local_experiment` takes experiment name and uploads
it to the server if its not already uploaded.

.. autofunction:: pypadre.app.padre_app.ExperimentApp.upload_local_experiment

In the following example we create experiment on local, authenticate and then upload it to server

.. literalinclude:: ../../tests/examples/example_upload_local_experiment.py

Download experiment
*******************
Experiment from server can be downloaded on local if its not already exists.
In the app :func:`pypadre.app.padre_app.ExperimentApp.download_remote_experiment` takes experiment id, name or url
and downloads it to the local if its not already exists.

.. autofunction:: pypadre.app.padre_app.ExperimentApp.download_remote_experiment


Versioning
**********


Live Logging
************

