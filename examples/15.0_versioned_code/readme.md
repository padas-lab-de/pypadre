## Versioned Code
The goal of versioned code is to keep track if there were any changes to the experiment source code in order
to create another execution (no changes) or another run (existing changes). We get the version code from
the ID (HASH) of the last commit object in the git repository.

### Example:
Let's say we executed our script **versioned_code.py**.

        from pypadre.examples.base_example import example_app
        import numpy as np
        from sklearn.decomposition import PCA
        from sklearn.datasets import load_iris
        from pypadre.pod.util.git_util import git_hash
        
        app = example_app()
        
        @app.dataset(name="iris", columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
                                           'petal width (cm)', 'class'], target_features='class')
        def dataset():
            data = load_iris().data
            target = load_iris().target.reshape(-1, 1)
            return np.append(data, target, axis=1)
        
        
        @app.parameter_map()
        def parameters():
            return {'SKLearnEstimator': {'parameters': {'SVC': {'C': [1.0]}, 'PCA': {'n_components': [3]}}}}
        
        
        @app.experiment(dataset=dataset, reference_git=__file__, parameters=parameters,
                        experiment_name="Iris SVC", project_name="Examples")
        def experiment():
            from sklearn.pipeline import Pipeline
            from sklearn.svm import SVC
            estimators = [('PCA', PCA()), ('SVC', SVC(probability=True))]
            return Pipeline(estimators)
        
        
        Code_Reference = persistent_hash((str(Path(__file__).parent),git_hash(str(Path(__file__).parent))))
        print('Versioned code git reference hash: {}'.format(Code_Reference))

The output of this script is:

![](screenshots/run1.png)

This code reference is at the same time the **ID** of the corresponding execution of the experiment.

* Screenshot #1 (The executions folder of the experiment) & (the runs folder of the experiment)

![](screenshots/executions1.png)   ![](screenshots/runs1.png)

As we can see the execution added has the code reference as the name of the folder.

If we re-run the same code in **versioned_code.py**, we will have the same execution but with another added run.

The output of the script is:

![](screenshots/run2.png)

* Screeshot #2 (the runs folder of the experiment)

![](screenshots/executions2.png) ![](screenshots/runs2.png)


### Changing the referenced code
After running this script let's say we changed the parameter_map :

    @app.parameter_map()
    def parameters():
        return {'SKLearnEstimator': {'parameters': {'SVC': {'C': [1.0]}, 'PCA': {'n_components': [4]}}}}


This would create a difference in the git repository and after commiting, we will have a new code version or HASH
which will create a new execution of the experiment instead of a new run since the source code creating the experiment
has changed.

The output of the changed code is :
 
![](screenshots/run3.png)

* Screeshot #3 (the executions folder of the experiment)

![](screenshots/executions3.png)

As we can see a new execution has been added with the new code reference.

![](screenshots/runs3.png)

While the runs of the previous execution stayed the same.
