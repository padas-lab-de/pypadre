from pypadre.eventhandler import trigger_event, assert_condition
from pypadre.base import MetadataEntity
from pypadre.core.model.split import Splitter, Split
import uuid


class Run(MetadataEntity):
    """
    A run is a single instantiation of an experiment with a definitive set of parameters.
    According to the experiment setup the pipeline/workflow will be executed
    """

    _results = []
    _hyperparameters = []
    _split_ids = None
    _id = None
    _metadata = None

    def __init__(self, experiment, workflow, **options):
        self.validate_parameters(experiment=experiment, workflow=workflow, options=options)
        self._experiment = experiment
        self._workflow = workflow
        self._keep_splits = options.pop("keep_splits", False)
        self._splits = []
        self._results = []
        self._metrics = []
        self._hyperparameters = []
        self._id = options.pop("run_id", None)
        self._split_ids = []
        super().__init__(self._id, **options)

        if self._id is None:
            self._id = uuid.uuid4()

    def do_splits(self):
        from copy import deepcopy
        # Fire run start event
        trigger_event('EVENT_START_RUN', run=self)

        # instantiate the splitter here based on the splitting configuration in options
        splitting = Splitter(self._experiment.dataset,  **self._metadata)
        for split, (train_idx, test_idx, val_idx) in enumerate(splitting.splits()):

            assert_condition(
                 condition=(self._experiment.validate.validate(train_idx, test_idx, val_idx, self._experiment.dataset)),
                 source=self,
                 message='Dataset Validation Failed')

            sp = Split(self, split, train_idx, val_idx, test_idx, **self._metadata)
            sp.execute()
            self._split_ids.append(str(sp.id)+'.split')
            if self._keep_splits is None:
                self._splits.append(sp)
            self._results.append(deepcopy(self._experiment.workflow.results))
            self._metrics.append(deepcopy(self._experiment.workflow.metrics))
            self._hyperparameters.append(deepcopy(self._experiment.workflow.hyperparameters))

        args = {'run': self}
        # Fire stop run  event
        trigger_event('EVENT_STOP_RUN', run=self)

    @property
    def experiment(self):
        return self._experiment

    @property
    def results(self):
        return self._results

    @results.setter
    def results(self, results):
        self._results = results

    @property
    def metrics(self):
        return self._metrics

    @metrics.setter
    def metrics(self, metrics):
        self._metrics = metrics

    @property
    def hyperparameters(self):
        return self._hyperparameters

    @property
    def workflow(self):
        return self._workflow

    @property
    def split_ids(self):
        return self._split_ids

    def __str__(self):
        s = []
        if self.id is not None:
            s.append("id:" + str(self.id))
        if self.name is not None and self.name != self.id:
            s.append("name:" + str(self.name))
        if len(s) == 0:
            return str(super())
        else:
            return "Run<" + ";".join(s) + ">"

    def validate_parameters(self, experiment, workflow, options):
        """
        This function validates the parameters passed to the constructor
        :param experiment: Experiment class object
        :param workflow: workflow of the experiment
        :param options: additional parameters
        :return:
        """
        from pypadre.core.model.experiment import Experiment
        assert_condition(condition=experiment is not None, source=self,
                         message="Experiment cannot be None")
        assert_condition(condition=isinstance(experiment, Experiment), source=self,
                         message="Parameter experiment is not an object of padre.core.Experiment")

        assert_condition(condition=workflow is not None, source=self,
                         message="Workflow cannot be none")
        assert_condition(condition=hasattr(workflow, 'fit') is True, source=self,
                         message='Workflow does not have a fit function')

        assert_condition(condition=options is not None, source=self,
                         message='Options parameter cannot be None')
        assert_condition(condition=isinstance(options, dict), source=self,
                         message='Options is not a dictionary type')

        assert_condition(condition=isinstance(options.get('keep_splits', True), bool), source=self,
                         message='keep_splits parameter is not of type bool')


