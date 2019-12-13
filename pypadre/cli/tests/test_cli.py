import os
import re
import shutil
import tempfile
import unittest

from click.testing import CliRunner

from pypadre.cli.pypadre import pypadre
from pypadre.pod.app import PadreConfig

SCRIPT= """from pypadre.core.model.pipeline.pipeline import DefaultPythonExperimentPipeline
from pypadre.binding.model.sklearn_binding import SKLearnPipeline
from pypadre.pod.app.padre_app import PadreAppFactory

app = PadreAppFactory.get(config)


@app.dataset(name="iris", columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
                                   'petal width (cm)', 'class'], target_features='class')
def data():
    from sklearn.datasets import load_iris
    import numpy as np
    data = load_iris().data
    target = load_iris().target.reshape(-1, 1)
    return np.append(data, target, axis=1)

@app.custom_splitter(name="Custom splitter", reference_git=path)
def splitter(dataset, **kwargs):
    import numpy as np
    idx = np.arange(dataset.size[0])
    cutoff = int(len(idx) / 2)
    return idx[:cutoff], idx[cutoff:], None

@app.parameter_map()
def params():
    return {'SKLearnEstimator': {'parameters': {'SVC': {'C': [1.0,2.0,3.0]}, 'PCA': {'n_components': [3,4]}}}}

@app.experiment(dataset=data, reference_git=path, parameters=params, splitting=splitter,
                experiment_name=experiment_name, project_name=project_name, ptype=SKLearnPipeline)
def main():
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    from sklearn.decomposition import PCA
    estimators = [('PCA', PCA(n_components=4)), ('SVC', SVC(probability=True))]
    return Pipeline(estimators)\n
"""

SCRIPT2= """from pypadre.core.model.pipeline.pipeline import DefaultPythonExperimentPipeline
from pypadre.binding.model.sklearn_binding import SKLearnPipeline
from pypadre.pod.app.padre_app import PadreAppFactory

app = PadreAppFactory.get(config)


@app.dataset(name="iris", columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
                                   'petal width (cm)', 'class'], target_features='class')
def data():
    from sklearn.datasets import load_iris
    import numpy as np
    data = load_iris().data
    target = load_iris().target.reshape(-1, 1)
    return np.append(data, target, axis=1)

@app.preprocessing(reference_git=path)
def preprocessing(dataset, **kwargs):
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    scaler = StandardScaler()
    scaler.fit(dataset.features())
    _features = scaler.transform(dataset.features())
    targets = dataset.targets()
    new_data = np.hstack((_features, targets))
    return new_data

@app.parameter_map()
def params():
    return {'SKLearnEstimator': {'parameters': {'SVC': {'C': [1.0,2.0,3.0]}, 'PCA': {'n_components': [3,4]}}}}

@app.experiment(dataset=data, reference_git=path, parameters=params, preprocessing_fn=preprocessing,
                experiment_name=experiment_name, project_name=project_name, ptype=SKLearnPipeline)
def main():
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    from sklearn.decomposition import PCA
    estimators = [('PCA', PCA()), ('SVC', SVC(probability=True))]
    return Pipeline(estimators)\n
"""


# noinspection PyMethodMayBeStatic
class PadreCli(unittest.TestCase):
    workspace_path = os.path.join(os.path.expanduser("~"), ".pypadre-test-cli")

    def __init__(self, *args, **kwargs):
        super(PadreCli, self).__init__(*args, **kwargs)
        config = PadreConfig(config_file=os.path.join(os.path.expanduser("~"), ".padre-test-cli.cfg"))
        config.set("backends", str([
            {
                "root_dir": os.path.join(os.path.expanduser("~"), ".pypadre-test-cli")
            }
        ]))

    def tearDown(self):
        # delete data content
        try:
            if os.path.exists(os.path.join(self.workspace_path, "datasets")):
                shutil.rmtree(self.workspace_path + "/datasets")
            if os.path.exists(os.path.join(self.workspace_path, "projects")):
                shutil.rmtree(self.workspace_path + "/projects")
            if os.path.exists(os.path.join(self.workspace_path, "code")):
                shutil.rmtree(self.workspace_path + "/code")
        except FileNotFoundError:
            pass

    def __del__(self):
        pass
        # delete configuration

    def test_dataset(self):
        runner = CliRunner()

        result = runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-test-cli.cfg"),
                                         'dataset', 'load', '-d'])
        assert result.exit_code == 0
        result = runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-test-cli.cfg"),
                                         'dataset', 'list'])
        assert '_boston' in result.output

        result = runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-test-cli.cfg"),
                                         'dataset', 'get',
                                         '_diabetes_dataset'])
        assert '_diabetes' in result.output

    def test_project(self):
        runner = CliRunner()

        result = runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-test-cli.cfg"),
                                         'project', 'create', '-n', 'Examples'])
        assert result.exit_code == 0

        result = runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-test-cli.cfg"),
                                         'project', 'list'])

        assert "Examples" in result.output

        result = runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-test-cli.cfg"),
                                         'project', 'select', 'Examples'])

        assert result.exit_code == 0

    def test_experiment(self):
        runner = CliRunner()

        runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-test-cli.cfg"),
                                'project', 'create', '-n', 'Examples'])

        temp_dir = tempfile.TemporaryDirectory()
        path = temp_dir.name + "/Experiment1.py"
        with open(path, mode="w") as f:
            f.write(SCRIPT)

        result = runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-test-cli.cfg"),
                                         'project', 'select', 'Examples'],
                               input="experiment execute --name Experiment1 --path {}\nn\nk".format(path))

        assert result.exit_code == 0
        assert 'Execution of the experiment is finished!' in result.output



        temp_dir_ = tempfile.TemporaryDirectory()
        path = temp_dir_.name + "/Experiment2.py"
        with open(path, mode="w") as f:
            f.write(SCRIPT2)

        result = runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-test-cli.cfg"),
                                         'project', 'select', 'Examples'],
                               input="experiment execute --name Experiment2 --path {}\nn\nk".format(path))

        assert result.exit_code == 0
        assert 'Execution of the experiment is finished!' in result.output



        result = runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-test-cli.cfg"),
                                         'experiment', 'list', '-c', 'id'])

        assert 'Experiment1' in result.output and 'Experiment2' in result.output

        import re
        ids = re.findall(r'\b\w+[-]\d+\b', result.output)

        result = runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-test-cli.cfg"),
                                         'experiment', 'compare', ids[0], ids[1]])

        temp_dir.cleanup()
        temp_dir_.cleanup()

        id = [s for s in result.output.split() if s.startswith('Experiment1-')].pop()

        result = runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-test-cli.cfg"),
                                         'experiment', 'get', id])

        assert id in result.output

        result = runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-test-cli.cfg"),
                                         'experiment', 'select', id])

        assert result.exit_code == 0

        temp_dir.cleanup()

    def test_execution(self):
        runner = CliRunner()
        runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-test-cli.cfg"),
                                'project', 'create', '-n', 'Examples'])

        temp_dir = tempfile.TemporaryDirectory()
        path = temp_dir.name + "/Experiment1.py"
        with open(path, mode="w") as f:
            f.write(SCRIPT)
        runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-test-cli.cfg"),
                                'project', 'select', 'Examples'],
                      input="experiment execute --name Experiment1 --path {}\nn\nk".format(path))

        with open(path, "a") as f:
            f.write("# Change added for testing")

        runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-test-cli.cfg"),
                                'project', 'select', 'Examples'],
                      input="experiment execute --name Experiment1 --path {}\nn\nk".format(path))


        result = runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-test-cli.cfg"),
                                         'execution', 'list', '-c', 'id'])

        assert result.exit_code == 0

        import re
        ids = re.findall(r'\b\d+[-]\d+\b', result.output)

        result = runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-test-cli.cfg"),
                                         'execution', 'get', ids[0]])

        assert ids[0] in result.output

        result = runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-test-cli.cfg"),
                                         'execution', 'compare', ids[0], ids[1]])

        assert result.exit_code == 0


        result = runner.invoke(pypadre, ['--config-file', os.path.join(os.path.expanduser("~"), ".padre-test-cli.cfg"),
                                         'execution', 'select', ids[0]])
        assert result.exit_code == 0

        temp_dir.cleanup()


if __name__ == '__main__':
    unittest.main()
