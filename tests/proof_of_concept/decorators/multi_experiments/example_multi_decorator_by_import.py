"""
This file shows an example on how to use PyPaDRE via decorators defining multipe experiments through an import

Note: it is a proof of concept now rather than a test.
"""
from tests.proof_of_concept.decorators.multi_experiments.ex1 import *
from tests.proof_of_concept.decorators.multi_experiments.ex2 import *


if __name__ == '__main__':
    exs = run()  # run the experiment and report
    for ex in exs:
        for r in ex.runs:
            print (ex.name+": "+str(r))
