from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np
from Cython.Build import cythonize

filename = 'cpython_nms'
full_filename = 'cpython_nms.pyx'

ext_modules = [Extension(filename, [full_filename],
                         language='c++',
                         extra_compile_args=['-O3', '-march=native', '-ffast-math', '/openmp'],
                         extra_link_args=['/openmp'])]

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules,
    include_dirs=[np.get_include()])
