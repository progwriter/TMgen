# coding=utf-8

from tmgen.tm cimport TrafficMatrix

cpdef TrafficMatrix modulated_gravity_tm(int num_pops, int num_tms,
                                         double mean_traffic,
                                         double pm_ratio=*, double t_ratio=*,
                                         double diurnal_freq=*,
                                         double spatial_variance=*,
                                         double temporal_variance=*)
cpdef TrafficMatrix uniform_tm(int num_pops, double low, double high,
                               int num_epochs=*)
cpdef TrafficMatrix exp_tm(int num_pops, double mean_traffic, int num_epochs=*)
cpdef TrafficMatrix random_gravity_tm(int num_pops, double mean_traffic)
cpdef TrafficMatrix spike_tm(int num_pops, int num_spikes, double mean_spike,
                             int num_epochs=*)
cpdef TrafficMatrix gravity_tm(populations, double total_traffic)
cpdef TrafficMatrix exact_tm(int num_pops, double val, int num_epochs=*)
cpdef TrafficMatrix lognormal_tm(int num_pops, double mean_traffic,
                                 double sigma=*, int num_epochs=*)
