from distutils.core import setup
from Cython.Build import cythonize


setup(
    ext_modules=cythonize("cython_methods.pyx")
)

# clear pycache
# >m -R __pycache__

# any time you make changes to one of the cython modules, it needs to be re-compiled
# >python setup.py build_ext --inplace
