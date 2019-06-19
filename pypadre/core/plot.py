import altair as alt


class Plot:
    """
    Class providing commands for managing visualizations.
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
        data = self._dataset.data
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

    def get_chart_from_json(self, visualisation):
        """
        Get altair Chart from json to render in notebook
        :param visualisation: Json specification of vega lite
        :returns: Altair Chart
        :rtype: <class 'altair.vegalite.v2.api.Chart'>
        """
        return alt.Chart.from_json(visualisation)

