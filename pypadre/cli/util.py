import os
import shutil
from click_shell import make_click_shell

from pypadre.pod.app.base_app import BaseEntityApp


def shorten_prompt_id(id):
    id = str(id)
    return (id[:5] + '..' + id[-5:]) if len(id) > 12 else id


def make_sub_shell(ctx, obj_name, obj, intro):
    wrapping_prompt = ctx.obj['prompt']
    prompt = ctx.obj['prompt'] + (obj_name[:3] if len(obj_name) > 3 else obj_name) + ": " + shorten_prompt_id(
        obj.id) + ' > '
    s = make_click_shell(ctx, prompt=prompt, intro=intro,
                         hist_file=os.path.join(os.path.expanduser('~'), '.click-pypadre-history'))
    ctx.obj['prompt'] = prompt
    ctx.obj[obj_name] = obj
    s.cmdloop()
    ctx.obj['prompt'] = wrapping_prompt
    del ctx.obj[obj_name]


def get_by_app(ctx, app: BaseEntityApp, id):
    objects = app.list({"id": id})
    if len(objects) == 0:
        print(app.model_clz.__name__ + " {0} not found!".format(id))
        return None
    if len(objects) > 1:
        print("Multiple matching entries of type " + app.model_clz.__name__ + " found!")
        _print_class_table(ctx, app.model_clz, objects)
        return None
    return objects.pop(0)


def _print_class_table(ctx, clz, *args, **kwargs):
    ctx.obj["pypadre-app"].print_tables(clz, *args, **kwargs)


def _create_experiment_file(path=None, file_name=None):
    if not os.path.exists(path):
        os.makedirs(path)
    _path = path + '/' + file_name + '.py'
    try:
        with open(_path,"w") as f:
            f.write(EXPERIMENT_TEMPLATE)

        return _path
    except Exception as e:
        raise ValueError(e)


EXPERIMENT_TEMPLATE = """ # This file is a template for defining your experiment pipeline and hyperparameters.
from pypadre.core.model.pipeline.pipeline import DefaultPythonExperimentPipeline
from pypadre.binding.model.sklearn_binding import SKLearnPipeline
from pypadre.pod.app.padre_app import PadreAppFactory

app = PadreAppFactory.get(config)

# Variables config, project_name, experiment_name and path are to be replaced automatically

# Add necessary imports inside the functions definition


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
def esitmator_config():
    # Hyperparameters for the estimator
    
    return {'param1':[value1,...],}

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

