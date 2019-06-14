"""
This example is to create and upload dataset scatter plot to server
For this example we will download iris dataset from openml and upload it to Padre server
then create scatter plots and upload specification to server
"""
from pypadre.app import p_app

p_app.authenticate("hmafnan", "test")  # Server user name and password here

oml_dataset_id = "61"  # Iris dataset
ds = p_app.remote_backend.datasets.load_oml_dataset(oml_dataset_id)
dataset_id = p_app.remote_backend.datasets.put(ds)

vis = ds.get_scatter_plot("petallength", "petalwidth", "Petal Length", "Petal Width")
p_app.remote_backend.datasets.put_visualisation(dataset_id, vis)