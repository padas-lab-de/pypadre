"""
The configuration-file 'mappings.json' is read during import of the module and stored statically inside the module.
"""

import json
import os

with open(os.path.join(os.path.dirname(__file__), "../parameters/mapping.json"), encoding='utf-8-sig') as f:
    algorithms = json.loads(f.read())['algorithms']

type_mappings = {}
name_mappings = {}

for alg in algorithms:
    name_mappings[alg['name']] = alg

    for k in alg['implementation']:
        type_mappings[alg['implementation'][k]] = (alg, k)