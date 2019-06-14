Visualisations
==============

Dataset visualization
---------------------
Scatter plot for the attributes of dataset can be created in the form of json specification(vega lite). For creating visualization
altair library is used.
To create scatter plot get_scatter_plot should be called on Dataset with dataset attributes of the users choice.

1. Example of creating scatter plot specification for Iris dataset and uploading it to the server.

.. literalinclude:: ../../tests/examples/example_create_and_upload_dataset_visualization.py

**Note:** pypadre.core.datasets.Dataset#get_scatter_plot takes x attribute and y attribute as required arguments for
which plot should be created.


