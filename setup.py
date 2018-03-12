from setuptools import setup, find_packages

setup(
    name='PyPaDRE - Python Client for PADAS Data Science Reproducibility Environment',
    version='0.1',
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
    # for click setup see http://click.pocoo.org/6/setuptools/
)
