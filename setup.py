import os

from pip.req import parse_requirements
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


def requirements(fname='requirements.txt'):
    return [str(r.req) for r in parse_requirements(abspath(fname))]


setup(name='deepcpg',
      version='1.0.0',
      description='Deep learning for predicting CpG methylation',
      long_description=read('README.rst'),
      author='Christof Angermueller',
      author_email='cangermueller@gmail.com',
      license = "BSD"
      url='https://github.com/cangermueller/deepcpg2',
      packages=['deepcpg'],
      install_requires=requirements(),
      )
