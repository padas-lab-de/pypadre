"""
Create and save visualization for scatter plot.

For this example Iris dataset will be loaded and saved on the local file system.
"""
import numpy as np
import os

from sklearn.datasets import load_iris

from pypadre.core import plot
from pypadre.examples.base_example import example_app

# create example app
app = example_app()


@app.dataset(name="iris",
             columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
                      'petal width (cm)', 'class'], target_features='class')
def dataset():
    data = load_iris().data
    target = load_iris().target.reshape(-1, 1)
    return np.append(data, target, axis=1)


ds = dataset
app.backends[0].dataset.put(ds, allow_overwrite=True)  # Save dataset locally


plt = plot.DataPlot(ds)
vis = plt.get_scatter_plot("sepal length (cm)",
                           "sepal width (cm)",
                           "Sepal Length",
                           "Sepal Width")

# Save visualization locally for above dataset
app.backends[0].dataset.put_visualization(vis,
                                          file_name="scatter_plot.json",
                                          base_path=os.path.join(app.backends[0].dataset.root_dir, ds.name))
