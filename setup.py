from setuptools import setup
import os

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(name='deepcpg',
      version='1.0.0',
      description='Deep learning for predicting CpG methylation',
      author='Christof Angermueller',
      author_email='cangermueller@gmail.com',
      license = "BSD"
      )
