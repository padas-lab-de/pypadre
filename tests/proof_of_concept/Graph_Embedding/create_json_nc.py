'''This file was based on sklearn official documentation https://scikit-learn.org/stable/user_guide.html'''
from copy import deepcopy
import json
import os

# The different strings present in the dictionaries are declared below
optional = "optional"
_type = "type"
path = "path"
_int = "int"
_tuple = "tuple"
_dict = "dict"
_bool = "bool"
_float = "float"
_list = "list"
_str = "str"
_tensor = "Tensor"
_iterable = "iterable"
default = "default"
params = "params"

#The different strategies are declared below
onevsrest = "ONEVSRESTCLASSIFIER"
onevsone = "ONEVSONECLASSIFIER"

#the different parameters of the strategies are declared below
n_jobs = "n_jobs"


n_jobs_dict = dict()
n_jobs_dict[_type] = [_int]
n_jobs_dict[optional] = True
n_jobs_dict[default] = None


#strategies
strategies_dict = dict()

#One vs rest classifier Definition
onevsrest_params = dict()
onevsrest_params[n_jobs] = deepcopy(n_jobs_dict)

onevsrest_dict = dict()
onevsrest_dict[path] = "sklearn.multiclass.OneVsRestClassifier"
onevsrest_dict[params] = deepcopy(onevsrest_params)

strategies_dict[onevsrest] = deepcopy(onevsrest_dict)

#One vs one Classifier Definition
onevsone_params = dict()
onevsone_params[n_jobs] = deepcopy(n_jobs_dict)

onevsone_dict = dict()
onevsone_dict[path] = "sklearn.multiclass.OneVsRestClassifier"
onevsone_dict[params] = deepcopy(onevsone_params)

strategies_dict[onevsone] = deepcopy(onevsone_dict)


#The different estimators
logisticregression = "LOGISTICREGRESSION"
linearregression = "LINEARREGRESSION"

#the different parameters of the estimators are decalred below
penalty_norm = "penalty_norm"
dual = "dual"
tolerance = "tolerance"
inverse_regularisation_strength = "inverse_regularisation_strength"
fit_intercept = "fit_intercept"
intercept_scaling = "intercept_scaling"
class_weight = "class_weight"
random_state = "random_state"
solver = "solver"
multi_class = "multi_class"
verbose = "verbose"
max_iterations = "max_iterations"
reuse_previous = "reuse_previous"
jobs = "jobs"

penalty_norm_dict = dict()
penalty_norm_dict[_type] = [_str]
penalty_norm_dict[optional] = False
penalty_norm_dict[default] = "l2"

dual_dict = dict()
dual_dict[_type] = [_bool]
dual_dict[optional] = False
dual_dict[default] = False

tolerance_dict = dict()
tolerance_dict[_type] = [_float]
tolerance_dict[optional] = False
tolerance_dict[default] = 0.0001

inverse_regularisation_strength_dict = dict()
inverse_regularisation_strength_dict[_type] = [_float]
inverse_regularisation_strength_dict[optional] = False
inverse_regularisation_strength_dict[default] = 0.1

fit_intercept_dict = dict()
fit_intercept_dict[_type] = [_bool]
fit_intercept_dict[optional] = False
fit_intercept_dict[default] = True

intercept_scaling_dict = dict()
intercept_scaling_dict[_type] = [_float]
intercept_scaling_dict[optional] = False
intercept_scaling_dict[default] = 1


class_weight_dict = dict()
class_weight_dict[_type] = [_dict, _str]
class_weight_dict[optional] = False
class_weight_dict[default] = None

random_state_dict = dict()
random_state_dict[_type] = [_int]
random_state_dict[optional] = True
random_state_dict[default] = None

solver_dict = dict()
solver_dict[_type] = [_str]
solver_dict[optional] = False
solver_dict[default] = "liblinear"

multi_class_dict = dict()
multi_class_dict[_type] = [_str]
multi_class_dict[optional] = False
multi_class_dict[default] = "ovr"

verbose_dict = dict()
verbose_dict[_type] = [_int]
verbose_dict[optional] = True
verbose_dict[default] = 0

max_iterations_dict = dict()
max_iterations_dict[_type] = [_int]
max_iterations_dict[optional] = False
max_iterations_dict[default] = 100

reuse_previous_dict = dict()
reuse_previous_dict[_type] = [_bool]
reuse_previous_dict[optional] = False
reuse_previous_dict[default] = False

jobs_dict = dict()
jobs_dict[_type] = [_int]
jobs_dict[optional] = False
jobs_dict[default] = 1


estimators_dict = dict()

#logistic Regression
logisticregression_params = dict()
logisticregression_params[penalty_norm] = deepcopy(penalty_norm_dict)
logisticregression_params[dual] = deepcopy(dual_dict)
logisticregression_params[tolerance] = deepcopy(tolerance_dict)
logisticregression_params[inverse_regularisation_strength] = deepcopy(inverse_regularisation_strength_dict)
logisticregression_params[fit_intercept] = deepcopy(fit_intercept_dict)
logisticregression_params[intercept_scaling] = deepcopy(intercept_scaling_dict)
logisticregression_params[class_weight] = deepcopy(class_weight_dict)
logisticregression_params[random_state] = deepcopy(random_state_dict)
logisticregression_params[solver] = deepcopy(solver_dict)
logisticregression_params[multi_class] = deepcopy(multi_class_dict)
logisticregression_params[verbose] = deepcopy(verbose_dict)
logisticregression_params[max_iterations] = deepcopy(max_iterations_dict)
logisticregression_params[reuse_previous] = deepcopy(reuse_previous_dict)
logisticregression_params[jobs] = deepcopy(jobs_dict)


logisticregression_dict = dict()
logisticregression_dict[path] = "sklearn.linear_model.logistic.LogisticRegression"
logisticregression_dict[params] = deepcopy(logisticregression_params)


estimators_dict[logisticregression] = deepcopy(logisticregression_dict)


#linear regression parameters
normalize = "normalize"
copy_X = "copy_X"


normalize_dict = dict()
normalize_dict[_type] = [_bool]
normalize_dict[optional] = True
normalize_dict[default] = False

copy_X_dict = dict()
copy_X_dict[_type] = [_bool]
copy_X_dict[optional] = True
copy_X_dict[default] = True


linearregression_params = dict()
linearregression_params[fit_intercept] = deepcopy(fit_intercept_dict)
linearregression_params[normalize] = deepcopy(normalize_dict)
linearregression_params[copy_X]  = deepcopy(copy_X_dict)


linearregression_dict = dict()
linearregression_dict[path] = "sklearn.linear_model.base.LinearRegression"
linearregression_dict[params] = deepcopy(linearregression_params)


estimators_dict[linearregression] = deepcopy(linearregression_dict)





#different metrics
f1_score = "F1_SCORE"
precision = "PRECISION"
recall = "RECALL"

#different parameters
y_true = "y_true"
y_pred = "y_pred"
labels = "labels"
pos_label = "pos_label"
average = "average"
sample_weight = "sample_weight"


y_true_dict = dict()
y_true_dict[_type] = [_iterable]
y_true_dict[optional] = False

y_pred_dict = dict()
y_pred_dict[_type] = [_iterable]
y_pred_dict[optional] = False

labels_dict = dict()
labels_dict[_type] = [_list]
labels_dict[optional] = True
labels_dict[default] = None


pos_label_dict =  dict()
pos_label_dict[_type] =  [_str, _int]
pos_label_dict[optional] = True
pos_label_dict[default] = 1

average_dict = dict()
average_dict[_type] = [_str]
average_dict[optional] = False
average_dict[default] = "binary"

sample_weight_dict = dict()
sample_weight_dict[_type] = [_iterable]
sample_weight_dict[optional] = True
sample_weight_dict[default] = None


#f1 _score metric
f1_score_params = dict()
f1_score_params[y_true] = deepcopy(y_true_dict)
f1_score_params[y_pred] = deepcopy(y_pred_dict)
f1_score_params[labels] = deepcopy(labels_dict)
f1_score_params[pos_label] = deepcopy(pos_label_dict)
f1_score_params[average] = deepcopy(average_dict)
f1_score_params[sample_weight] = deepcopy(sample_weight_dict)

f1_score_dict = dict()
f1_score_dict[path] = "sklearn.metrics.f1_score"
f1_score_dict[params] = deepcopy(f1_score_params)


#metrics dict
metrics_dict = dict()

metrics_dict[f1_score] = deepcopy(f1_score_dict)



#
strategies = "strategies"
estimators = "estimators"
metrics = "metrics"

framework_dict = dict()
framework_dict[strategies] = deepcopy(strategies_dict)
framework_dict[estimators] = deepcopy(estimators_dict)
framework_dict[metrics] = deepcopy(metrics_dict)


cwd = os.getcwd()
print(cwd)
with open('mappings_nc.json', 'w') as fp:
    json.dump(framework_dict, fp)