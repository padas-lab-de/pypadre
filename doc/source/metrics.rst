**************
CompareMetrics
**************
=========
Summary
=========
This class is used to compare the results of different experiments. It could be based on a single experiment,
in which case the runs of only that experiment are considered or multiple experiments where the runs of all the experiments
input are considered



Steps for comparing runs through CLI
------------------------------------
Use the command "**compare_metrics**" to compare runs with the parameters.

#. path: Path of the experiments to be compared.
#. query: Conditions based on which the results are displayed. The available options are

    * all: Displays all the runs
    * names of estimators: Displays those runs which have these estimators. Multiple estimators can be given separated by a *comma*. Example, 'isomap,principal component analysis'
    * estimator.param.value: Displays those runs which have a particular value for a particular parameter. Multiple estimator.param.values can be given separated by a *comma*. Example, 'principal component analysis.num_components.5, principal component analysis.num_components.6'

#. metrics: The metrics that need to be displayed to the user. Example "mean_error"
Sample execution: compare_metrics --path "~/pypadre/experiments/Test Experiment PCA Linear.ex" --query "principal component analysis" --metrics "mean_error"






