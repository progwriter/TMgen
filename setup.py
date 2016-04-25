# coding=utf-8


from distutils.core import setup

import numpy
from Cython.Build import cythonize

setup(
    name='tmgen',
    version='0.1',

    author='Victor Heorhiadi',
    author_email='victor@cs.unc.edu',

    package_dir={'': 'src'},
    packages=['tmgen'],
    url='https://github.com/progwriter/tmgen',
    requires=['numpy', 'cython'],
    include_dirs=[numpy.get_include()],
    ext_modules=cythonize("src/tmgen/**/*.pyx")
)
