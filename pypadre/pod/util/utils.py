import ctypes
import inspect
import os
import pip
from pypadre.binding.model.sklearn_evaluator import SKLearnEvaluator

from pypadre.core.model.split.splitter import default_split
from pypadre.core.model.pipeline.components.components import DefaultSplitComponent
from pypadre.binding.model.sklearn_estimator import SKLearnEstimator
from pypadre.core.printing.util.print_util import get_default_table
from pypadre.core.model.pipeline.pipeline import Pipeline
from pypadre.pod.constants import RESOURCE_DIRECTORY_PATH


class ResourceDirectory:

    def create_directory(self):
        # TODO create a corresponding configuration object. look up best practices
        data_dir = os.path.expanduser(RESOURCE_DIRECTORY_PATH)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        return data_dir


FILE_ATTRIBUTE_HIDDEN = 0x02


def write_hidden(file_name, data):
    """
    Cross platform hidden file writer.
    see https://stackoverflow.com/questions/25432139/python-cross-platform-hidden-file
    """
    # For *nix add a '.' prefix.
    prefix = '.' if os.name != 'nt' else ''
    file_name = prefix + file_name

    # Write file.
    with open(file_name, 'w') as f:
        f.write(data)

    # For windows set file attribute.
    if os.name == 'nt':
        ret = ctypes.windll.kernel32.SetFileAttributesW(file_name,
                                                        FILE_ATTRIBUTE_HIDDEN)
        if not ret:  # There was an error.
            raise ctypes.WinError()


def compare_metas(meta1: dict, meta2: dict, on_keys: list):
    keys = meta1.keys() & meta2.keys() & set(on_keys)
    diff_dict = dict()
    for k in keys:
        diff_dict[k] = (meta1.get(k), meta2.get(k))

    return diff_dict


def compare_pipelines(pipeline1: Pipeline, pipeline2: Pipeline):
    components = {"preprocessor": (pipeline1.preprocessor, pipeline2.preprocessor),
                  "splitter": (pipeline1.splitter, pipeline2.splitter),
                  "estimator": (pipeline1.estimator, pipeline2.estimator),
                  "evaluator": (pipeline1.evaluator, pipeline2.evaluator)}

    # TODO get the metadata, source code diff for each component
    diff_dict = dict()
    for component in components:
        source = []
        name = []
        type = []
        reference = []
        for comp in components.get(component):
            if comp:
                name.append(comp.name)
                reference.append(comp.reference.id)
                type.append(comp.__class__.__name__)
                if isinstance(comp, SKLearnEstimator):
                    source.append(str(comp.pipeline))
                elif isinstance(comp, DefaultSplitComponent):
                    source.append("builtin method split in pypadre.core.model.split.splitter")
                elif isinstance(comp, SKLearnEvaluator):
                    source.append("builtin method in pypadre.binding.model.sklearn_evaluator")
                else:
                    source.append(str(inspect.getsource(comp.code.fn)))
            else:
                name.append("None")
                reference.append("None")
                type.append("None")
                source.append("None")
        source = [ s.split("\n") for s in source]
        diff_dict[component] = {
            "name": (name[0],name[1]),
            "reference ID": (reference[0],reference[1]),
            "type": (type[0], type[1]),
            "source code": (source[0],source[1])}
    return diff_dict


def diff_to_table(dict_: dict, columns=None):
    table = get_default_table()
    if columns:
        table.column_headers = columns
    else:
        table.column_headers = ["attribute","Element 1", "Element 2"]
    for key, item in dict_.items():
        if isinstance(item, tuple):
            table.append_row([str(key), str(item[0]), str(item[1])])
        elif isinstance(item, dict):
            table.append_row([str(key),str("-"),str("-"), str("-")])
            for k, (v,w) in item.items():
                if any(isinstance(el, list) for el in [v,w]):
                    #padding
                    if len(v) >= len(w):
                        w = w + ["~" for i in range(len(v)-1)]
                    else:
                        v = v + ["~" for i in range(len(w)-1)]
                    table.append_row([str("~"), str(k), str(v[0]), str(w[0])])
                    for i in range(1,len(v)):
                        table.append_row([str("~"), str("~"), str(v[i]), str(w[i])])
                else:
                    table.append_row([str("~"), str(k), str(v), str(w)])
    if len(table)==0:
        table.append_row([str("-") for x in table.column_headers])

    return table

