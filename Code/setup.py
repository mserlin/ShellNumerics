from distutils.core import setup
from Cython.Build import cythonize

setup(name ="FastJacobian", ext_modules=cythonize('FastJacobian.pyx'),)