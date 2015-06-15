from setuptools import setup
import os

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(name='predict',
      version='0.0.1',
      description='Single-cell methylation state prediction',
      author='Christof Angermueller',
      author_email='cangermueller@gmail.com',
      license = "BSD"
      )
