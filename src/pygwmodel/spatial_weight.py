from typing import Optional
from enum import IntEnum
from .py_spatial_weight import SpatialWeight as SpatialWeightBind
from .py_spatial_weight import BandwidthWeight as BandwidthWeightBind



class Distance:

    def as_args(self) -> tuple:
        raise NotImplemented()


class CRSDistance(Distance):
    is_geographic = False

    def __init__(self, is_geographic: bool=False) -> None:
        super().__init__()
        self.is_geographic = is_geographic
    
    def as_args(self):
        return (self.is_geographic,)


class Weight:

    def as_args(self) -> tuple:
        raise NotImplemented()


class BandwidthWeight(Weight):

    class Kernel(IntEnum):
        Gaussian = BandwidthWeightBind.Gaussian
        Exponential = BandwidthWeightBind.Exponential
        Bisquare = BandwidthWeightBind.Bisquare
        Tricube = BandwidthWeightBind.Tricube
        Boxcar = BandwidthWeightBind.Boxcar

    bandwidth: Optional[float] = None
    adaptive: bool = False
    kernel: Kernel = Kernel.Gaussian

    def __init__(self, bandwidth: Optional[float]=None, adaptive: bool=False, kernel: Kernel=Kernel.Gaussian) -> None:
        super().__init__()
        self.bandwidth = bandwidth
        self.adaptive = adaptive
        self.kernel = kernel
    
    def as_args(self) -> tuple:
        return (self.bandwidth, self.adaptive, self.kernel.value)


class SpatialWeight:
    
    @staticmethod
    def create(distance: Distance, weight: Weight) -> None:
        sw = SpatialWeightBind()
        ''' Set distance
        '''
        if isinstance(distance, CRSDistance):
            sw.set_distance_crs(*distance.as_args())
        else:
            raise TypeError("Distance type not supported")
        ''' Set weight
        '''
        if isinstance(weight, BandwidthWeight):
            sw.set_weight_bandwidth(*weight.as_args())
        else:
            raise TypeError("Weight type not supported")
        return sw

