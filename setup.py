import os

from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(name='deepcpg',
      version='1.0.0',
      description='Deep learning for predicting CpG methylation',
      long_description=read('README.md'),
      author='Christof Angermueller',
      author_email='cangermueller@gmail.com',
      license="BSD",
      url='https://github.com/cangermueller/deepcpg2',
      packages=['deepcpg'],
      setup_requires=['scipy'],
      install_requires=['argparse',
                        'numpy',
                        'h5py',
                        'pandas',
                        'pytest',
                        'scikit-learn',
                        'keras',
                        'matplotlib',
                        'seaborn']
      )
