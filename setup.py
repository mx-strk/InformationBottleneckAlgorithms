from Cython.Build import cythonize
import numpy
from Cython.Distutils import build_ext
from setuptools import setup
from setuptools.extension import Extension

cmdclass = {}
extensions = [Extension("*", ["information_bottleneck/tools/*.pyx"],
                        include_dirs=[numpy.get_include()])]
cmdclass.update({ 'build_ext': build_ext })

setup(name='ib_base',
      version='1.0',
      author='Maximilian Stark, Jan Lewandowsky',
      author_email='maximilian.stark@tuhh.de',
      python_requires='>=3',
      packages=['information_bottleneck.tools','information_bottleneck.information_bottleneck_algorithms'],
      ext_package='information_bottleneck',
      ext_modules = cythonize(extensions),
      cmdclass = cmdclass,
      package_data={'information_bottleneck.information_bottleneck_algorithms':
                        ['*.cl']},
      )