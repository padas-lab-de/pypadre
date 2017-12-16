from padre.schema import Attribute, ExperimentSchema, ListAttribute, SelectSchema

PCA_Schema = \
    {

    }

SVC_Schema = \
    {
        "C" : Attribute("C", "Penalty parameter C of the error term.", False, float),
        "kernel" : Attribute("kernel", "Specifies the kernel type to be used in the algorithm. It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’ or ‘precomputed’.", False, str),
        "degree": Attribute("deg", "Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.", False, int)
    }

KNeighbors_Schema = \
    {

    }

def selectStepSchema(data):

    steps = {
        "PCA" : PCA_Schema,
        "SVC" : SVC_Schema,
        "KNeighbors" : KNeighbors_Schema
    }

    return steps[data["algorithm"]]

PipelineSchema = ExperimentSchema(
    {
        "steps":ListAttribute("steps", "The steps", False, [
            {
                "doc":Attribute("doc", "Docstring", True, str),
                "algorithm":Attribute("algorithm", "The name of the used algorithm", False, str)
            },
            SelectSchema(selectStepSchema)
        ]),
        "doc":Attribute("doc", "Docstring", True, str)
    }
)
