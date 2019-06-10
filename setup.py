from distutils.core import setup
from Cython.Build import cythonize
import numpy


setup(
    ext_modules=cythonize(["toolbelt/*.pyx", "toolbelt/feature_extraction/*.pyx"]), #"toolbelt/cluster/*.pyx"]),
    include_dirs=[numpy.get_include()]
)

# clear pycache
# >m -R __pycache__

# any time you make changes to one of the cython modules, it needs to be re-compiled
# >python setup.py build_ext --inplace
