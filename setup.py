# from sphinx.setup_command import BuildDoc
import re

from setuptools import setup, find_packages

# cmdclass = {'build_sphinx': BuildDoc}

NAMEFILE = "pypadre/_name.py"
verstrline = open(NAMEFILE, "rt").read()
VSRE = r"^__name__ = ['\"]([^'\"]*)['\"]"
result = re.search(VSRE, verstrline, re.M)
if result:
    name = result.group(1)
else:
    raise RuntimeError("Unable to find name string in %s." % (NAMEFILE,))

VERSIONFILE = "pypadre/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
result = re.search(VSRE, verstrline, re.M)
if result:
    version = result.group(1)
    release = version
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

print('VERSION:{version}'.format(version=version))

with open('requirements.txt', 'r') as f:
    required = f.read()

install_requires = required

setup(
    name=name,
    version=version,
    # TODO switch to scm version
    # use_scm_version=True,
    # setup_requires=['setuptools_scm'],
    packages=['pypadre'] + find_packages(exclude=["tests", "tests.*", "*.tests.*", "*.tests"]),
    package_dir={'pypadre': 'pypadre'},
    include_package_data=True,
    url='https://padre-lab.eu',
    license='GPL',
    author='Michael Granitzer',
    author_email='michael.granitzer@uni-passau.de',
    description='PaDRe aims to solve problems about reproducibility',
    entry_points='''
       [console_scripts]
       pypadre=pypadre.cli.pypadre:pypadre
   ''',
    install_requires=install_requires,
    command_options={
        'build_sphinx': {
            'project': ('setup.py', name),
            'version': ('setup.py', version),
            'release': ('setup.py', release)}},
    # for click setup see http://click.pocoo.org/6/setuptools/
)
