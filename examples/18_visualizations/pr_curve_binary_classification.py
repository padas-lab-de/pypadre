"""
Create and save visualization for precision recall curve for binary classification.

For this example a new experiment will be executed and saved on the local system.
"""
import uuid

from pypadre.pod.importing.dataset.dataset_import import SKLearnLoader

from pypadre.core import plot
from pypadre.examples.base_example import example_app

# create example app
app = example_app()

loader = SKLearnLoader()
breast_cancer = loader.load("sklearn", utility="load_breast_cancer")


name = str(uuid.uuid4())[0:10]


@app.experiment(dataset=breast_cancer, reference_git=__file__,
                experiment_name="Binary Precision Recall Visualization Example " + name,
                seed=1, project_name="Examples")
def experiment():
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    estimators = [('SVC', SVC(probability=True))]
    return Pipeline(estimators)


parameter_dict = {'SVC': {'C': [0.1, 0.2]}}
experiment.execute(parameters=
                   {'SKLearnEvaluator': {'write_results': True}, 'SKLearnEstimator': {'parameters': parameter_dict}})

run_id = experiment.executions[0].run[0].id
output_id = ""

for output_pipline in app._pipeline_output_app.list():
    if output_pipline.parent.id == run_id:
        output_id = output_pipline.id

plt = plot.ExperimentPlot(
    experiment.project.name,
    experiment.name,
    str(experiment.executions[0].id),
    run_id,
    output_id,
    base_path=app.backends[0].root_dir)

# Save visualization for the current experiment
vis = plt.plot_pr_curve("Precision Recall")
app.backends[0].dataset.put_visualization(vis,
                                          file_name="binary_precision_recall.json",
                                          base_path=plt.split_path())

