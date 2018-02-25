from glob import glob
import os

from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(name='deepcpg',
      version='1.0.6',
      description='Deep learning for predicting CpG methylation',
      long_description=read('README.rst'),
      author='Christof Angermueller',
      author_email='cangermueller@gmail.com',
      license="MIT",
      url='https://github.com/cangermueller/deepcpg',
      packages=find_packages(),
      scripts=glob('./scripts/*.py'),
      install_requires=['h5py',
                        'argparse',
                        'scikit-learn',
                        'scipy',
                        'pandas',
                        'numpy',
                        'pytest',
                        'keras',
                        'matplotlib',
                        'seaborn'],
      keywords=['Deep learning',
                'Deep neural networks',
                'Epigenetics',
                'DNA methylation',
                'Single cells'],
      classifiers=['Development Status :: 5 - Production/Stable',
                   'Environment :: Console',
                   'Intended Audience :: Science/Research',
                   'License :: OSI Approved :: MIT License',
                   'Natural Language :: English',
                   'Programming Language :: Python',
                   'Programming Language :: Python :: 2',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.3',
                   'Programming Language :: Python :: 3.4',
                   'Programming Language :: Python :: 3.5',
                   'Topic :: Scientific/Engineering :: Artificial Intelligence',
                   'Topic :: Scientific/Engineering :: Bio-Informatics',
                   ]
      )
