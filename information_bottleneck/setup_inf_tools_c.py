from distutils.core import setup

import numpy
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("tools/inf_theory_tools_c.pyx"),
    include_dirs=[numpy.get_include()]
)

setup(
    ext_modules=cythonize("tools/basic_tools_c.pyx"),
    include_dirs=[numpy.get_include()]
)