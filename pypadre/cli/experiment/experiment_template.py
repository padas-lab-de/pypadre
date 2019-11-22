import os
from pypadre.pod.app import PadreConfig
from pypadre.pod.app.padre_app import PadreAppFactory
from pypadre.core.model.pipeline.pipeline import DefaultPythonExperimentPipeline
from pypadre.core.util.utils import unpack

#Add necessary imports

#####
# Configuration of the padre app.
config_path = os.path.join(os.path.expanduser("~"), ".Config-file.cfg")
workspace_path = os.path.join(os.path.expanduser("~"), ".pypadre-folder")

config = PadreConfig(config_file=config_path)
config.set("backends", str([
{
    "root_dir": workspace_path
}
]))
config.save()
app = PadreAppFactory.get(config)

#####
@app.dataset(name="", columns=[], target_features='')
def dataset():
    ##TODO Write code for the dataset loading
    return 


#####
@app.parameter_map()
def config():
    #TODO Hyperparameters for the estimator
    raise NotImplementedError()
    return {}


@app.custom_splitter(name="Custom splitter", reference_git=__file__)
def custom_splitter(dataset, **kwargs):
    # TODO Custom code for data splitting, OPTIONAL. 
    # Return the training, testing, and validation indices 
    raise NotImplementedError()
    return train_idx, test_idx, val_idx/None


@app.estimator(config=config, reference_git=__file__)
def estimator(X_train, y_train, *args, **kwargs):
    #TODO define your training algorithm in here. 
    #Return the fitted model
    raise NotImplementedError()
    return model


@app.evaluator(task_type="Classification", reference_git=__file__)
def evaluator(model, X_test, *args, **kwargs):
    #TODO write your evaluator that returns the predicted values and the probabilities if possible. 
    raise NotImplementedError()
    return predicted_values, probabilities


@app.experiment(dataset=dataset, reference_git=__file__, splitting=custom_splitter,
                estimator=estimator, evaluator=evaluator,
                experiment_name="", project_name="", ptype=DefaultPythonExperimentPipeline)
def experiment():
    return
