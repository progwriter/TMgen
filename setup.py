# coding=utf-8

from distutils.core import setup
# from setuptools import setup
# from setuptools.extension import Extension

try:
    import numpy
    from Cython.Build import cythonize
except ImportError as e:
    print ('TMgen requires numpy and cython to be installed!')
    raise e

setup(
    name='tmgen',
    version='0.1.1',

    author='Victor Heorhiadi',
    author_email='victor@cs.unc.edu',

    package_dir={'': 'src'},
    packages=['tmgen'],
    url='https://github.com/progwriter/tmgen',
    requires=['numpy', 'cython', 'six'],
    include_dirs=[numpy.get_include()],
    ext_modules=cythonize("src/tmgen/**/*.pyx"),
    package_data={
        'tmgen': ['*.pxd'],
    }
)
