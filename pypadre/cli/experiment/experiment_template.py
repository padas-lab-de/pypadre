from pypadre.core.model.pipeline.pipeline import DefaultPythonExperimentPipeline
from pypadre.binding.model.sklearn_binding import SKLearnPipeline
from pypadre.core.util.utils import unpack

# Add necessary imports inside the functions definition

#####
"""
This file is a template for defining your experiment pipeline and hyperparameters.
You can use The SkLearnPipeline binding which takes a sklearn pipeline or a custom pipeline where you define your own 
estimator and evaluator.
"""


@app.dataset(name="", columns=[], target_features='')
def dataset():
    ##TODO Write code for the dataset loading
    return data





@app.custom_splitter(name="Custom splitter", reference_git=__file__)
def custom_splitter(dataset, **kwargs):
    # TODO Custom code for data splitting (see examples/13_custom_splitting), OPTIONAL you can use the default splitter
    #  (see examples/12_splitting)
    # Return the training, testing, and validation indices 
    raise NotImplementedError()
    return train_idx, test_idx, val_idx / None


@app.preprocessing(reference_git=__file__, store=True)
def preprocessing(dataset, **kwargs):
    _features = dataset.features()
    _targets = dataset.targets()
    # TODO do the preprocessing of the data (see examples/11_preprocessing)
    raise NotImplementedError()
    return transformed_data


# When using a custom pipline you can build your own estimator and evaluator like the following (see examples/14_custom_pipeline)

"""
@app.parameter_map()
def config():
    # Hyperparameters for the estimator
    raise NotImplementedError()
    return {}

@app.estimator(config=config, reference_git=__file__)
def estimator(X_train, y_train, *args, **kwargs):
    # TODO define your training algorithm in here.
    # Return the fitted model
    raise NotImplementedError()
    return model


@app.evaluator(task_type="Classification", reference_git=__file__)
def evaluator(model, X_test, *args, **kwargs):
    # TODO write your evaluator that returns the predicted values and the probabilities if possible.
    raise NotImplementedError()
    return predicted_values, probabilities


@app.experiment(dataset=dataset, reference_git=__file__, splitting=custom_splitter,
                preprocessing_fn=preprocessing,
                estimator=estimator, evaluator=evaluator,
                experiment_name=experiment, project_name=project,
                ptype=DefaultPythonExperimentPipeline)
def experiment():
    return
"""

# Or you can use the sklearn binding and work with estimators of sklearn.
"""
@app.parameter_map()
def parameters():
    # Define the hyperparameters of the estimator
    return {'SKLearnEstimator': {'parameters': {'Estimator1': {},..., 'EstimatorN': {} }}}

@app.experiment(dataset=dataset, reference_git=__file__, splitting=custom_splitter, parameters=parameters,
                preprocessing_fn=preprocessing,
                experiment_name=experiment, project_name=project,
                ptype=SkLearnPipeline)
def experiment():
    from sklearn.pipeline import Pipeline
    
    estimators = []
    return Pipeline(estimators)
"""