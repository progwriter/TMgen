from numpy.typing import ArrayLike

from tmgen.tm import TrafficMatrix

def modulated_gravity_tm(
    num_nodes: int,
    num_tms: int,
    mean_traffic: float,
    pm_ratio: float = 1.5,
    t_ratio: float = 0.75,
    diurnal_freq: float = ...,
    spatial_variance: float = 100,
    temporal_variance: float = 0.01,
) -> TrafficMatrix: ...
def random_gravity_tm(num_nodes: int, mean_traffic: float) -> TrafficMatrix: ...
def gravity_tm(populations: ArrayLike, total_traffic: float) -> TrafficMatrix: ...
def uniform_tm(
    num_nodes: int, low: float, high: float, num_epochs: int = 1
) -> TrafficMatrix: ...
def exp_tm(
    num_nodes: int, mean_traffic: float, num_epochs: int = 1
) -> TrafficMatrix: ...
def spike_tm(
    num_nodes: int, num_spikes: int, mean_spike: float, num_epochs: int = 1
) -> TrafficMatrix: ...
def lognormal_tm(
    num_nodes: int, mean_traffic: float, sigma: float = 1, num_epochs: int = 1
) -> TrafficMatrix: ...
def exact_tm(num_nodes: int, val: float, num_epochs: int = 1) -> TrafficMatrix: ...
