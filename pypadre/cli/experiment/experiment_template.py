from pypadre.core.model.pipeline.pipeline import DefaultPythonExperimentPipeline
from pypadre.binding.model.sklearn_binding import SKLearnPipeline
from pypadre.pod.app.padre_app import PadreAppFactory

app = PadreAppFactory.get(config)

# Add necessary imports inside the functions definition
"""
This file is a template for defining your experiment pipeline and hyperparameters.


#### You can use The SkLearnPipeline binding which takes a sklearn pipeline or a custom pipeline where you define your own 
estimator and evaluator.



@app.dataset(name="", columns=[], target_features='')
def data():
    # Write code for the dataset loading
    
    return data

@app.custom_splitter(name="Custom splitter", reference_git=__file__)
def splitter(dataset, **kwargs):
    # Custom code for data splitting (see examples/13_custom_splitting), OPTIONAL you can use the default splitter
    #  (see examples/12_splitting)
    # Return the training, testing, and validation indices 
    
    return train_idx, test_idx, (val_idx or None)


@app.preprocessing(reference_git=__file__, store=True)
def preprocessing_fn(dataset, **kwargs):
    _features = dataset.features()
    _targets = dataset.targets()
    # Do the preprocessing of the data (see examples/11_preprocessing)
    
    return transformed_data


# When using a custom pipline you can build your own estimator and evaluator like the following (see examples/14_custom_pipeline)


@app.parameter_map()
def config():
    # Hyperparameters for the estimator
    
    return {}

@app.estimator(config=config, reference_git=path)
def custom_estimator(X_train, y_train, *args, **kwargs):
    # TODO define your training algorithm in here.
    # Return the fitted model
    
    return model

@app.evaluator(task_type="Classification", reference_git=path)
def custom_evaluator(model, X_test, *args, **kwargs):
    # TODO write your evaluator that returns the predicted values and the probabilities if possible.
    raise NotImplementedError()
    return predicted_values, probabilities

@app.experiment(dataset=data, reference_git=path, splitting=splitter,
                preprocessing_fn=preprocessing_fn,
                estimator=custom_estimator, evaluator=custom_evaluator,
                experiment_name=experiment_name, project_name=project_name,
                ptype=DefaultPythonExperimentPipeline)
def main():
    return


# Or you can use the sklearn binding and work with estimators of sklearn (see examples/.

@app.parameter_map()
def parameters():
    # Define the hyperparameters of the estimator
    
    return {'SKLearnEstimator': {'parameters': {'Estimator1': {},..., 'EstimatorN': {} }}}

@app.experiment(dataset=data, reference_git=path, splitting=splitter, parameters=parameters,
                preprocessing_fn=preprocessing_fn,
                experiment_name=experiment_name, project_name=project_name,
                ptype=SkLearnPipeline)
def main():
    from sklearn.pipeline import Pipeline
    # Imports here
    estimators = []
    return Pipeline(estimators) 
"""