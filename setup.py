from setuptools import setup, find_packages
from sphinx.setup_command import BuildDoc

cmdclass = {'build_sphinx': BuildDoc}

name = 'PyPaDRE - Python Client for PADAS Data Science Reproducibility Environment'
version = '0.0.1'
release = '0.0.1'

setup(
    name=name,
    version=version,
    packages=find_packages(),
    url='',
    license='GPL',
    author='Michael Granitzer',
    author_email='michael.granitzer@uni-passau.de',
    description='',
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