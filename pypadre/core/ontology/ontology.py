from importlib import resources

from owlready2 import get_ontology

with resources.open_binary("pypadre.core.resources.ontology", "padre_workflow.owl") as f:
    padre_ontology = get_ontology("http://www.padre-lab.eu/onto/padre_workflow").load(fileobj=f)
