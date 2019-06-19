"""
This example is to create and upload dataset scatter plot to server
For this example we will download iris dataset from openml and upload
it to Padre server then create scatter plots and upload specification
to server

Note: To run this example local server should be up and running
"""
from pypadre.app import p_app
from pypadre.core import plot

p_app.authenticate("hmafnan", "test")  # user name and password here

oml_dataset_id = "61"  # Iris dataset
ds = p_app.remote_backend.datasets.load_oml_dataset(oml_dataset_id)
dataset_id = p_app.remote_backend.datasets.put(ds)
plt = plot.Plot(ds)
vis = plt.get_scatter_plot("petallength", "petalwidth", "Petal Length", "Petal Width")
p_app.remote_backend.datasets.put_visualisation(dataset_id, vis)