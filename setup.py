import os

from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(name='deepcpg',
      version='1.0.1',
      description='Deep learning for predicting CpG methylation',
      long_description=read('README.rst'),
      author='Christof Angermueller',
      author_email='cangermueller@gmail.com',
      license="MIT",
      url='https://github.com/cangermueller/deepcpg',
      packages=find_packages(),
      install_requires=['h5py',
                        'argparse',
                        'scikit-learn',
                        'scipy',
                        'numpy',
                        'pandas',
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
                   'Programming Language :: Python :: 3',
                   'Topic :: Scientific/Engineering :: Artificial Intelligence',
                   'Topic :: Scientific/Engineering :: Bio-Informatics',
                   ]
      )
