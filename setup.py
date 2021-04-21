from setuptools import setup, find_packages
import os
from Cython.Build import cythonize
import numpy

__location__ = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(__location__, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
setup(
    name='Toolbelt',
    version='0.0.4',
    description='Various Objects and Methods to make life easier.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='AJ Wilson',
    author_email='aj.wilson08@gmail.com',
    classifiers=['Development Status :: 3 - Alpha',
                 'Intended Audience :: Data Science',
                 'Topic :: Data Science & Statistics',
                 'Programming Language :: Python :: 3.6+'],
    keywords='Statistics Data Science Tools',
    packages=find_packages(),
    python_requires='>=3.6.8',
    install_requires=['Cython>=0.29', "pandas>=0.24", "numpy>= 1.16.2", "matplotlib>=3",'seaborn>=0.9', "scipy>=1.2",
                      "networkx>=2", "scikit-learn>=0.20"],
    project_urls={'Source': 'github.com'},
    ext_modules=cythonize(["toolbelt/feature_extraction/*.pyx",
                           "toolbelt/utils/*.pyx",
                           "toolbelt/trees/*.pyx",
                           "toolbelt/stats/*.pyx"],
                          ),
    include_dirs=[numpy.get_include()]
)

# any time you make changes to one of the cython modules, it needs to be re-compiled
# >python setup.py build_ext --inplace
