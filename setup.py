from setuptools import setup, find_packages
from sphinx.setup_command import BuildDoc
import os

cmdclass = {'build_sphinx': BuildDoc}

# See https://medium.com/@pypripackages/using-gitlab-pipelines-to-deploy-python-packages-in-production-and-staging-environments-8ab7dc979274

name = 'pypadre'
#if os.environ.get('CI_COMMIT_TAG'):
#    version = os.environ['CI_COMMIT_TAG']
#else:
#    version = os.environ['CI_JOB_ID']
version = os.environ.get('VERSION')
release = os.environ.get('VERSION')

print('VERSION:{version}'.format(version=version))

setup(
    name=name,
    version=version,
    packages=['padre']+find_packages(exclude=["tests", "tests.*"]),
    package_dir={'padre': 'padre'},
    package_data={'padre': ['res/mapping/pytorch.json', 'res/mapping/mapping.json',
                            '/core/wrappers/wrapper_mappings/mappings_torch.json']},
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
