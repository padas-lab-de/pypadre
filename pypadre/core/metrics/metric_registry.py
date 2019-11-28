from typing import List, Set

from networkx import DiGraph, descendants

from pypadre.core.metrics.metrics import MetricProviderMixin, Metric
from pypadre.core.model.computation.computation import Computation
from pypadre.core.model.generic.i_model_mixins import LoggableMixin


class MetricRegistry(LoggableMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    registered_providers = DiGraph()
    providers_by_consumption = {}

    def get_provider_by_name(self, name):
        pass

    def add_providers(self, *providers):
        for provider in providers:
            self.add_provider(provider)

    def add_provider(self, provider: MetricProviderMixin):
        if provider in self.registered_providers.nodes:
            self.send_warn("Measure already defined. Omitted adding it to the measure service: " + str(provider))
        else:
            # Todo more sophisticated adding
            self.registered_providers.add_node(provider)
            for node in self.registered_providers.nodes:
                if provider.consumes in node.provides:
                    self.registered_providers.add_edge(node, provider)
                if node.consumes in provider.provides:
                    self.registered_providers.add_edge(provider, node)
            if not hasattr(self.providers_by_consumption, provider.consumes):
                self.providers_by_consumption[provider.consumes] = [provider]
            else:
                self.providers_by_consumption[provider.consumes].append(provider)

    def initial_providers(self, computation: Computation):
        entries = []
        for node in self.registered_providers.nodes:
            if node.consumes == computation.format:
                entries.append(node)
        return entries

    def available_providers(self, computation: Computation) -> Set[MetricProviderMixin]:
        metric_providers = set()
        for node in self.initial_providers(computation):
            metric_providers.add(node)
            for provider in descendants(self.registered_providers, node):
                metric_providers.add(provider)
        return metric_providers

    def calculate_measures(self, computation: Computation, providers=None, **kwargs) -> List[Metric]:
        if providers is None:
            providers = self.available_providers(computation)

        else:
            from networkx.algorithms.simple_paths import all_simple_paths
            execution_paths = []
            for initial_provider in self.initial_providers(computation):
                for provider in providers:

                    # If the required metric is the initial metric itself
                    if provider == initial_provider:
                        execution_paths.append([provider])
                        continue

                    paths = all_simple_paths(self.registered_providers, source=initial_provider,
                                              target=provider)
                    for path in paths:
                        execution_paths.append(path)

            return self.execute_paths(execution_paths, computation, **kwargs)



        provider_history = set()
        results = []

        # each provider can currently only be ran once
        for provider in self.initial_providers(computation):
            if provider in providers and provider not in provider_history:
                measured = provider.execute(computation=computation, **kwargs)
                if isinstance(measured, Metric):
                    results.append(measured)
                else:
                    results = results + measured
                provider_history.add(provider)
                results = self._calculate_helper(computation, provider, measured, providers, provider_history, results, **kwargs)
        return results

    def _calculate_helper(self, computation: Computation, current_provider, current_data, providers, provider_history, results, **kwargs):
        # each provider can currently only be ran once
        for provider in self.registered_providers.successors(current_provider):
            if provider in providers and provider not in provider_history:
                measured = provider.execute(computation=computation, data=current_data, **kwargs)
                if isinstance(measured, Metric):
                    results.append(measured)
                else:
                    results = results + measured
                provider_history.add(provider)
                results = self._calculate_helper(computation, provider, measured, providers, provider_history, results, **kwargs)
        return results

    def execute_paths(self, execution_paths, computation, **kwargs):
        results = []

        for execution_path in execution_paths:
            measured = []
            initial_node = True
            for provider in execution_path:

                if initial_node is True:
                    initial_node = False
                    measured = provider.execute(computation=computation, **kwargs)
                    continue

                measured = provider.execute(computation=computation, data=measured)

            if isinstance(measured, Metric):
                results.append(measured)
            else:
                results = results + measured

        return results





metric_registry = MetricRegistry()
