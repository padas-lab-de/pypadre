from setuptools import setup, find_packages
from sphinx.setup_command import BuildDoc
import re
import os

cmdclass = {'build_sphinx': BuildDoc}

name = 'padre'
VERSIONFILE="padre/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
result = re.search(VSRE, verstrline, re.M)
if result:
    version = result.group(1)
    release = version
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

print('VERSION:{version}'.format(version=version))

install_requires =["msgpack_numpy==0.4.1",
                   "requests==2.13.0",
                   "numpy==1.13.3",
                   "scipy==1.0.0",
                   "click==6.7",
                   "scikit_learn==0.19.1",
                   "pandas-profiling==1.4.1",
                   "liac-arff==2.2.1",
                   "protobuf==3.6.1",
                   "networkx==2.1",
                   "beautifultable==0.5.2",
                   "openml==0.7.0",
                   "pandas==0.23.3",
                   "requests-toolbelt==0.8.0",
                   "mock==2.0.0",
                   "beautifulsoup4==4.6.3",
                   "Deprecated==1.2.4",
                   "pymitter==0.2.3",
                   "altair==2.4.1",
                   "vega==1.4.0",
                   "vega_datasets==0.7.0",
                   "notebook==5.7.4",
                   "seaborn==0.9.0",
                   "matplotlib==3.0.0",
                   "Sphinx==2.0.1"]

setup(
    name=name,
    version=version,
    packages=['padre']+find_packages(exclude=["tests", "tests.*"]),
    package_dir={'padre': 'padre'},
    package_data={'padre': ['padre/res/mapping/pytorch.json', 'padre/res/mapping/mapping.json',
                            'padre/core/wrappers/wrapper_mappings/mappings_torch.json']},
    #packages=find_packages(),
    include_package_data=True,
    url='https://padre-lab.eu',
    license='GPL',
    author='Michael Granitzer',
    author_email='michael.granitzer@uni-passau.de',
    description='PaDRe aims to solve problems about reproducibility',
    entry_points='''
       [console_scripts]
       padre=padre.app.padre_cli:main
   ''',
    install_requires=install_requires,
   command_options={
        'build_sphinx': {
            'project': ('setup.py', name),
            'version': ('setup.py', version),
            'release': ('setup.py', release)}},
    # for click setup see http://click.pocoo.org/6/setuptools/
)

# todo add requirements txt for tests only
