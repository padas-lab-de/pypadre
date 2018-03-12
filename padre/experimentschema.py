from padre.schema import Attribute, Schema, ListAttribute, SelectSchema, AlgorithmSchema


PipelineSchema = Schema(
    {
        "steps": ListAttribute("steps", "The steps", False,[
            {
                "doc": Attribute("doc", "Docstring", True, str),
                "algorithm": Attribute("algorithm", "The name of the used algorithm", False, str)
            },
            AlgorithmSchema()
        ]),
        "doc": Attribute("doc", "Docstring", True, str)
    }
)
