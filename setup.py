from distutils.core import setup
import subprocess
from distutils.command.build import build as DistutilsBuild
from setuptools  import find_packages
import sys
from os import chdir

VERSION = '0.1.0'
from distutils.util import get_platform

try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            #Using compiled code that depends on the plataform
            self.plat_name = get_platform()
            _bdist_wheel.finalize_options(self)
            self.root_is_pure = True

except ImportError:
    bdist_wheel = None

class MyBuild(DistutilsBuild):
    def run(self):
      chdir('spclustering')
      print('running makefile:')
      if subprocess.call(['make','makelib']) != 0:
            sys.exit(-1)
      chdir('..')

long_description = open("README.md").read()

setup(name='spclustering',
      version=VERSION,
      description='Python Wrapper for Superparamagnetic Clustering',
      long_description=long_description,
      long_description_content_type="text/markdown",
      author='Fernando Julian Chaure',
      author_email='fchaure@fi.uba.ar',
      url='https://github.com/ferchaure/SPC/',
      packages=['spclustering'],
      cmdclass={#'install': MyInstall
      'build': MyBuild,'bdist_wheel': bdist_wheel
      },
      package_data={"": ["*.c","*.h","Makefile",'spclib.so']},
      install_requires=['numpy','matplotlib'],
      keywords=['clustering'],
          classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
      ]
     )
