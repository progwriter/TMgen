# coding: utf-8
# cython: profile=True
# filename: mgm.py
from __future__ import print_function
import tmgen

"""
    Generate a Modulated Gravity traffic matrix
"""

# 20 nodes, 100 traffic matrices, mean number of flows between nodes is 10000.
print(tmgen.models.modulated_gravity_tm(20, 100, 10000))
