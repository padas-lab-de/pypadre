from setuptools import setup, find_packages
# from sphinx.setup_command import BuildDoc
import re

# cmdclass = {'build_sphinx': BuildDoc}

name = 'pypadre'
VERSIONFILE="pypadre/_version.py"
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
    packages=['pypadre']+find_packages(exclude=["tests", "tests.*"]),
    package_dir={'pypadre': 'pypadre'},
    package_data={'pypadre': ['pypadre/res/mapping/pytorch.json', 'pypadre/res/mapping/mapping.json',
                              'pypadre/core/wrappers/wrapper_mappings/mappings_torch.json']},
    #packages=find_packages(),
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

# todo add requirements txt for tests only
