from setuptools import setup, find_packages
from sphinx.setup_command import BuildDoc
import re
import os

cmdclass = {'build_sphinx': BuildDoc}

# See https://medium.com/@pypripackages/using-gitlab-pipelines-to-deploy-python-packages-in-production-and-staging-environments-8ab7dc979274
version = os.environ.get('VERSION')
release = os.environ.get('VERSION')

name = 'pypadre'
#if os.environ.get('CI_COMMIT_TAG'):
#    version = os.environ['CI_COMMIT_TAG']
#else:
#    version = os.environ['CI_JOB_ID']
VERSIONFILE="padre/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
result = re.search(VSRE, verstrline, re.M)
if result:
    version=result.group(1)
    release=version
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

print('VERSION:{version}'.format(version=version))

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
   command_options={
        'build_sphinx': {
            'project': ('setup.py', name),
            'version': ('setup.py', version),
            'release': ('setup.py', release)}},
    # for click setup see http://click.pocoo.org/6/setuptools/
)

# todo add requirements txt for tests only
