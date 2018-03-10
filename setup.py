from distutils.core import setup

setup(
    name='c.PyPaDRE',
    version='0.1',
    packages=['padre', 'padre.padre-server', 'tests'],
    url='',
    license='GPL',
    author='Michael Granitzer',
    author_email='',
    description='',
    entry_points='''
       [console_scripts]
       yourscript=yourscript:cli
   ''',
    # for click setup see http://click.pocoo.org/6/setuptools/
)
