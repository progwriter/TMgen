from __future__ import print_function
import sys

import numpy

sys.path.append("../src/")
import pyximport
pyximport.install(setup_args={'include_dirs': [numpy.get_include()]})
import tmgen

print(tmgen.simple_generator())
