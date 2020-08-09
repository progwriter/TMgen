# coding=utf-8

from setuptools import setup, find_packages
from setuptools.extension import Extension

try:
    import numpy
    from Cython.Build import cythonize
except ImportError as e:
    print('TMgen requires numpy and cython to be installed!')
    raise e

extensions = [
    Extension(
        'tmgen.hmc',
        ['src/tmgen/hmc.pyx'],
        include_dirs=[numpy.get_include()]
    ),
    Extension(
        'tmgen.models',
        ['src/tmgen/models.pyx'],
        include_dirs=[numpy.get_include()]
    ),
    Extension(
        'tmgen.tm',
        ['src/tmgen/tm.pyx'],
        include_dirs=[numpy.get_include()]
    )
]

setup(
    name='tmgen',
    version='0.1.2',
    description='Library for network traffic matrix generation',
    keywords=['network', 'traffic', 'matrix'],

    author='Victor Heorhiadi',
    author_email='victor@cs.unc.edu',
    license='MIT',

    package_dir={'': 'src'},
    packages=find_packages('src'),
    url='https://github.com/progwriter/TMgen',
    install_requires=['numpy', 'cython', 'six'],
    extras_require={
        'plotting': ['matplotlib', 'seaborn'],
    },
    tests_require=['pytest', 'flake8'],
    setup_requires=['pytest-runner'],
    ext_modules=cythonize(extensions),
    package_data={
        'tmgen': ['*.pxd'],
    },
    include_dirs=[numpy.get_include()],
    entry_points={
        'console_scripts': [
            'tmditg = tmgen.inject.ditg_injector:main'
        ],
    },

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',

        'Topic :: System :: Networking',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ]

)
