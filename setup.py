from setuptools import setup, find_packages
from sphinx.setup_command import BuildDoc

cmdclass = {'build_sphinx': BuildDoc}

name = 'pypadre'
version = '0.0.1'
release = '0.0.1'

setup(
    name=name,
    version=version,
    packages=['padre']+find_packages(exclude=["tests"]),
    package_dir={'padre': 'padre'},
    package_data={'padre': ['res/mapping/pytorch.json', 'res/mapping/mapping.json',
                            'core/wrappers/wrapper_mappings/mappings_torch.json',
                            'protobuffer/binaries/1kb_mixed_dataframe.protobinV1']},
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
