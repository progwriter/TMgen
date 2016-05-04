# coding=utf-8
from tmgen.tm cimport TrafficMatrix
import numpy
cimport numpy
from cpython cimport bool

cpdef TrafficMatrix modulated_gravity_tm(int num_pops, int num_tms,
                                         double mean_traffic,
                                         double pm_ratio= *, double t_ratio= *,
                                         double diurnal_freq= *,
                                         double spatial_variance= *,
                                         double temporal_variance= *)
cpdef tuple _modulated_gravity(numpy.ndarray mean_row,
                               numpy.ndarray mean_col,
                               numpy.ndarray modulated_total,
                               double sigmasq,
                               double sigmasq_temporal= *,
                               bool only_gravity= *)
cpdef numpy.ndarray _peak_mean_cycle(double freq, double n, double mean,
                                     double peak_to_mean,
                                     double trough_to_mean= *)
cpdef TrafficMatrix random_gravity_tm(int num_pops, double mean_traffic,
                                      double spatial_variance)