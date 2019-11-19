from pypadre.core.model.dataset.dataset import Transformation
from pypadre.core.util.utils import unpack
from pypadre.examples.base_example import example_app
import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from pypadre.binding.model.sklearn_binding import SKLearnPipeline


