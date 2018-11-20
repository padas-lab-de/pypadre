
from tkinter import filedialog
from padre.metrics import CompareMetrics
from padre.metrics import ReevaluationMetrics

import numpy as np
import pandas as pd

def main():
    # Load the results folder
    dir_path = filedialog.askdirectory(initialdir="~/.pypadre/experiments", title="Select Experiment Directory")
    # It could either be experiments in a directory or multiple experiments
    dir_list = list()
    if len(dir_path) == 0:
        return
    dir_list.append(dir_path)
    metrics = CompareMetrics(dir_path=dir_list)
    metrics.show_metrics()

    recompute_metrics = ReevaluationMetrics(dir_list)
    dir_path = filedialog.askdirectory(initialdir="~/.pypadre/experiments", title="Select Experiment Directory")
    dir_list.append(dir_path)
    recompute_metrics.get_split_directories()
    recompute_metrics.recompute_metrics()

    # pd_frame = ex.analyse_runs(run_query, [performance_measures], options)

    metrics.analyze_runs(['all'])
    df = metrics.display_results()
    print(df)
    metrics.analyze_runs(
        ['principal component analysis.num_components.4', 'principal component analysis.num_components.5'])
    df = metrics.display_results()
    print(df)

    metrics.analyze_runs(['principal component analysis.num_components.4'], ['mean_error'])
    df = metrics.display_results()
    print(df)


if __name__ == '__main__':
    main()
