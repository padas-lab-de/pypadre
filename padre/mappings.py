"""
The configuration-file 'mappings.json' is read during import of the module and stored statically inside the module.
"""

import json


with open("../padre/parameters/mapping.json") as f:
    # print(f.read())
    algorithms = json.loads(f.read()[3:])['algorithms']

type_mappings = {}
name_mappings = {}

for alg in algorithms:
    name_mappings[alg['name']] = alg

    for k in alg['implementation']:
        type_mappings[alg['implementation'][k]] = (alg, k)