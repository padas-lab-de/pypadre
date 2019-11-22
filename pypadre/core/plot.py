"""
Plots are implemented with Altair which is Python implementation of Vega-Lite
"""
import json
import os

import altair as alt
import numpy as np
import pandas as pd

from sklearn import metrics

import collections

class Plot:
    """
    Class providing commands for managing visualizations.
    """
    def get_chart_from_json(self, visualisation):
        """
        Get altair Chart from json to render in notebook
        :param visualisation: Json specification of vega lite
        :returns: Altair Chart
        :rtype: <class 'altair.vegalite.v2.api.Chart'>
        """
        return alt.Chart.from_json(visualisation)


class DataPlot(Plot):
    """
    Class providing commands for creating visualizations for Data sets.
    """
    def __init__(self, dataset):
        self._dataset = dataset

    def get_scatter_plot(self, x_attr, y_attr, x_title=None, y_title=None):
        """
        Get scatter plot of vega lite specification

        :param x_attr: first data set attribute
        :type x_attr: str
        :param y_attr: Second data set attribute
        :type y_attr: str
        :param x_title: Title for x attribute
        :type x_title: str
        :param y_title: Title for y attribute
        :type y_title: str
        :returns: Json Vega lite specification
        """
        data = pd.DataFrame(self._dataset.data(), columns=[attr["name"] for attr in self._dataset.attributes])
        target = self._dataset.get_target_attribute()
        if x_title is None:
            x_title = x_attr[0].upper() + x_attr[1:]
        if y_title is None:
            y_title = y_attr[0].upper() + y_attr[1:]
        chart = alt.Chart(data).mark_point().encode(
            x=alt.X(x_attr, title=x_title),
            y=alt.Y(y_attr, title=y_title),
            color=target + ":N"
        ).properties(
            title="%s dataset visualization for %s and %s" % (self._dataset.name, x_attr, y_attr)).interactive()
        return chart.to_json()

    def plot_correlation_matrix(self, title="Feature Correlation Matrix"):
        """Plot pearson correlation matrix between features.

        if number of features are less than 30 then show text inside matrix otherwise just show colors
        Supports numpy and pandas formats

        :param title: Title of the chart
        :return: Returns vega lite specification of the chart.
        """
        mx = pd.DataFrame(self._dataset.data(), columns=[attr["name"] for attr in self._dataset.attributes]
                          ).corr('pearson')
        total_features = mx.shape[1]
        x, y = np.meshgrid(range(0, total_features), range(0, total_features))
        source = pd.DataFrame(
            {'x': x.ravel(), 'y': y.ravel(), 'z': np.around(np.array(mx).ravel(), decimals=1)})
        # Adjust width and height of the chart
        if total_features <= 15:
            width, height = 500, 450
        else:
            width, height = 650, 550

        base = alt.Chart(source).encode(
            alt.X('x:O',
                  title='Features'
                  ),
            alt.Y('y:O', title='Features')
        ).properties(width=width, height=height)

        chart = base.mark_rect().encode(
            alt.Color('z:Q', title='Range', scale=alt.Scale(scheme='redblue'))
        ).properties(title=title)
        if total_features <= 30:
            text = base.mark_text(baseline='middle').encode(
                text='z:Q',
                color=alt.condition(
                    'datum.z > 0.5 || datum.z < 0.0',
                    alt.value('white'),
                    alt.value('black')
                )
            )
            chart = chart + text
        return chart.to_json()

    def plot_class_balance(self, title='Class Balance'):
        """Plot class balance chart.

        Treats classification data as nominal type and regression data as quantitative type.
        Supports numpy and pandas formats

        :param title: Title of the chart
        :return: Returns vega lite specification of the chart.
        """
        df = pd.DataFrame(self._dataset.data(), columns=[attr["name"] for attr in self._dataset.attributes])
        target = self._dataset.get_target_attribute()
        if df[target].nunique() <= 15:  # Treat as classification data
            chart = alt.Chart(df).mark_bar().encode(
                x=alt.X('count()', title='Total Records'),
                y=alt.Y(target+':N', title='Class Labels'),
                color=alt.Color(target + ':N', title='Labels')).properties(title=title)
        else:  # Treat as regression data
            chart = alt.Chart(df).mark_bar().encode(
                x=alt.X('count()', title='Range Count'),
                y=alt.Y(target + ':Q', title='Data Points'),
                color=alt.Color(target + ':Q', title='Range')).properties(title=title)
        return chart.to_json()


class ExperimentPlot(Plot):
    """
    Class providing commands for creating visualizations for experimental results.
    """
    def __init__(self, project_name, experiment_name, execution_id, run_id, split_id, base_path="~/.pypadre"):
        self._run_id = run_id
        self._split_id = split_id
        self._experiment_path = os.path.join(os.path.expanduser(base_path), *["projects",
                                                                              project_name,
                                                                              "experiments",
                                                                              experiment_name,
                                                                              "executions",
                                                                              execution_id])

    def split_path(self):
        split_path = os.path.join(self._experiment_path, *["runs", self._run_id, "output", self._split_id])
        if os.path.exists(split_path):
            return split_path
        else:
            raise NotADirectoryError("Split path does not exists")

    def plot_confusion_matrix(self, title='Confusion Matrix'):
        """Plot confusion matrix and return its json specification of vega-lite

        :param title: Title for the chart
        :returns: Json specification of Vega-Lite
        """
        split_path = self.split_path()
        with open(os.path.join(split_path, "metrics.json")) as f:
            metrics = json.loads(f.read())
        cm = np.array(metrics[list(metrics.keys())[0]][0])
        max_value = cm.max()
        x, y = np.meshgrid(range(0, cm.shape[0]), range(0, cm.shape[0]))
        source = pd.DataFrame({'x': x.ravel(), 'y': y.ravel(), 'z': cm.ravel()})
        base = alt.Chart(source).encode(
            alt.X('x:O', title='Predicted label'),
            alt.Y('y:O', title='True label')
        ).properties(width=400, height=400)

        heatmap = base.mark_rect().encode(
            alt.Color('z:Q', title='Value Range', scale=alt.Scale(scheme='purplebluegreen'))
        ).properties(title=title)

        text = base.mark_text(baseline='middle').encode(
            text='z:Q',
            color=alt.condition(
                alt.datum.z < max_value / 3,
                alt.value('black'),
                alt.value('white')
            )
        )

        chart = heatmap + text
        return chart.to_json()

    def plot_pr_curve(self, title='Precision Recall Curve'):
        """Plot precision recall curve and return its json specification of vega-lite

        Plots curves for Binary and multi class classification

        :param title: Title for the chart
        :returns: Json specification of Vega-Lite
        """
        split_path = self.split_path()
        with open(os.path.join(split_path, "results.json"), 'r') as f:
            results = json.loads(f.read())
        probabilities = list()
        truths = list()
        for k, predictions in results["predictions"].items():
            truths.append(predictions["truth"])
            probabilities.append(predictions["probabilities"])
        if len(probabilities[0]) > 2:  # Multi class classification
            # Use binary approach that is treat each class as 1 and all other classes as 0
            targets = set(truths)
            precisions = np.array([])
            recalls = np.array([])
            colors = np.array([])
            for target_label in targets:
                label_truths = [1.0 if truth == target_label else 0.0 for truth in truths]
                score = [x[int(target_label)] for x in probabilities]
                precision, recall, thresholds = metrics.precision_recall_curve(np.array(label_truths), np.array(score))
                precisions = np.concatenate([precisions, precision])
                recalls = np.concatenate([recalls, recall])
                colors = np.concatenate([colors, np.full((1, recall.size), int(target_label)).flatten()])
            data = pd.DataFrame({
                "x": recalls,
                "y": precisions,
                "color": colors
            })
            chart = alt.Chart(data).mark_line(
                interpolate='step-after',
            ).encode(
                x=alt.X('x', title="Recall"),
                y=alt.Y('y', title="Precision"),
                color=alt.Color('color:N', title="Labels")
            ).properties(title=title)
        else:  # Binary classification
            score = [x[-1] for x in probabilities]
            precision, recall, thresholds = metrics.precision_recall_curve(truths, score)
            data = pd.DataFrame({
                "x": recall,
                "y": precision
            })
            chart = alt.Chart(data).mark_area(
                color="lightblue",
                interpolate='step-after',
                line=True
            ).encode(
                x=alt.X('x', title="Recall"),
                y=alt.Y('y', title="Precision")
            ).properties(title=title)
        return chart.to_json()

    def plot_roc_curve(self, title='ROC Curve'):
        """Plot ROC curve and return its json specification of vega-lite

        Plots curves for Binary and multi class classification

        :param title: Title for the chart
        :returns: Json specification of Vega-Lite
        """
        split_path = self.split_path()
        with open(os.path.join(split_path, "results.json"), 'r') as f:
            results = json.loads(f.read())
        probabilities = list()
        truths = list()
        for k, predictions in results["predictions"].items():
            truths.append(predictions["truth"])
            probabilities.append(predictions["probabilities"])
        if len(probabilities[0]) > 2:  # Multi class classification
            # Use binary approach that is treat each class as 1 and all other classes as 0
            targets = set(truths)
            tprs = np.array([])
            fprs = np.array([])
            colors = np.array([])
            for target_label in targets:
                label_truths = [1.0 if truth == target_label else 0.0 for truth in truths]
                score = [x[int(target_label)] for x in probabilities]
                fpr, tpr, thresholds = metrics.roc_curve(np.array(label_truths), np.array(score))
                tprs = np.concatenate([tprs, tpr])
                fprs = np.concatenate([fprs, fpr])
                colors = np.concatenate([colors, np.full((1, fpr.size), int(target_label)).flatten()])
            data = pd.DataFrame({
                "x": fprs,
                "y": tprs,
                "color": colors
            })
            chart = alt.Chart(data).mark_line(
                interpolate='step-after',
            ).encode(
                x=alt.X('x', title="False Positive Rate"),
                y=alt.Y('y', title="True Positive Rate"),
                color=alt.Color('color:N', title="Labels")
            ).properties(title=title)
        else:  # Binary classification
            score = [x[-1] for x in probabilities]
            fpr, tpr, thresholds = metrics.roc_curve(truths, score)
            data = pd.DataFrame({
                "x": fpr,
                "y": tpr
            })
            chart = alt.Chart(data).mark_area(
                color="lightblue",
                interpolate='step-after',
                line=True
            ).encode(
                x=alt.X('x', title="False Positive Rate"),
                y=alt.Y('y', title="True Positive Rate")
            ).properties(title=title)
        return chart.to_json()



