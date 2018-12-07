"""
The configuration-file 'mappings.json' is read during import of the module and stored statically inside the module.
"""

import json
import os

type_mappings = {}
name_mappings = {}
version_mappings = {}
# TODO: Currently hard coded, but later should be read from the library tag in the mapping file
#supported_frameworks = ['scikit-learn', 'pytorch']

path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../res/mapping"))
mapping_files = os.listdir(path)

# For every file verify whether it is a json file and then add to the current type_mappings and name_mappings
for file in mapping_files:
    if os.path.isfile(os.path.join(path,file)):
        try:
            with open(os.path.join(path,file), encoding='utf-8-sig') as f:
                data = json.loads(f.read())
                algorithms = data['algorithms']
                metadata = data['metadata']
                version_mappings[metadata['library']] = metadata

            for alg in algorithms:
                name_mappings[alg['name']] = alg

                for k in alg['implementation']:
                    type_mappings[alg['implementation'][k]] = (alg, k)

        except:
            raise ValueError('Error when parsing JSON file {name}'.format(name=file))

supported_frameworks = list(version_mappings.keys())