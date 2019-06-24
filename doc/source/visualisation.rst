Visualisations
==============

Dataset visualization
---------------------

Scatter Plot
************

Scatter plot for the attributes of dataset can be created in the form of json specification(vega lite). For creating visualization
altair library is used.
To create scatter plot :func:`pypadre.core.plot.Plot.get_scatter_plot` function is provided in the plot module which should be
called with dataset attributes of the users choice.

.. autofunction:: pypadre.core.plot.Plot.get_scatter_plot

**Note:** :func:`pypadre.core.plot.Plot.get_scatter_plot` takes x attribute and y attribute as required arguments for
which plot should be created.

Following example is to create scatter plot specification for Iris dataset and upload it to the server.

.. literalinclude:: ../../tests/examples/example_create_and_upload_dataset_visualization.py




