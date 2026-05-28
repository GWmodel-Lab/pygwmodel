from typing import Optional
from ._spatial_weight import _SpatialWeight
from ._spatial_weight import _BandwidthWeight


class Distance:
    """Abstract base class for distance metrics."""

    def as_args(self) -> tuple:
        raise NotImplemented()


class CRSDistance(Distance):
    """Coordinate Reference System (Euclidean / Great-circle) distance.

    Computes straight-line (projected) or great-circle (geographic)
    distances between coordinate pairs.

    Parameters
    ----------
    is_geographic : bool, optional
        If True, use great-circle distance; otherwise use
        Euclidean distance (default False).
    """

    def __init__(self, is_geographic: bool = False) -> None:
        super().__init__()
        self.is_geographic = is_geographic

    def as_args(self):
        return (self.is_geographic,)


class CRSSTDistance(Distance):
    """Spatio-temporal distance for GTWR models.

    A distance metric that combines spatial distance with temporal
    distance via a ratio parameter :math:`\\lambda`.  When
    :math:`\\lambda = 0` the distance is purely temporal; when
    :math:`\\lambda = 1` it is purely spatial.

    Parameters
    ----------
    is_geographic : bool, optional
        Whether coordinates are geographic (default False).
    lambda_ : float, optional
        Spatio-temporal ratio in [0, 1] (default 0.5).
    """

    def __init__(self, is_geographic: bool = False,
                 lambda_: float = 0.5) -> None:
        super().__init__()
        self.is_geographic = is_geographic
        self.lambda_ = lambda_

    def as_args(self):
        return (self.is_geographic, self.lambda_)


class Weight:
    """Abstract base class for spatial weighting schemes."""

    def as_args(self) -> tuple:
        raise NotImplemented()


class BandwidthWeight(Weight):
    """Bandwidth-based spatial weighting with kernel function.

    Defines the scale of the spatial kernel.  An adaptive
    bandwidth uses the *k*-nearest neighbours while a fixed
    bandwidth uses a distance threshold.

    Parameters
    ----------
    bandwidth : float, optional
        Bandwidth size.  For adaptive=True this is the number
        of nearest neighbours; for adaptive=False this is a
        distance threshold.  Use None when automatic selection
        is desired.
    adaptive : bool, optional
        Use adaptive (k-NN) bandwidth (default False).
    kernel : BandwidthKernelType, optional
        Kernel function (default Gaussian).

    Attributes
    ----------
    Kernel : BandwidthKernelType
        Enumeration of available kernel functions:
        Gaussian, Exponential, Bisquare, Tricube, Boxcar.
    """

    Kernel = _BandwidthWeight.BandwidthKernelType

    bandwidth: Optional[float] = None
    adaptive: bool = False
    kernel: Kernel = Kernel.Gaussian

    def __init__(self, bandwidth: Optional[float] = None,
                 adaptive: bool = False,
                 kernel: Kernel = Kernel.Gaussian) -> None:
        super().__init__()
        self.bandwidth = bandwidth
        self.adaptive = adaptive
        self.kernel = kernel

    def as_args(self) -> tuple:
        return (self.bandwidth, self.adaptive, self.kernel.value)


class SpatialWeight:
    """Factory for creating spatial-weight configurations.

    Combines a distance metric with a weighting scheme into a
    single object used by GW models.

    Use the :meth:`create` static method to produce instances.
    """

    @staticmethod
    def create(distance: Distance, weight: Weight):
        """Create a spatial-weight from distance and weight.

        Parameters
        ----------
        distance : Distance
            A distance metric (CRSDistance or CRSSTDistance).
        weight : Weight
            A weighting scheme (BandwidthWeight).

        Returns
        -------
        _SpatialWeight
            Internal C++ spatial-weight object.
        """
        sw = _SpatialWeight()
        if isinstance(distance, CRSSTDistance):
            sw.set_distance_crst(*distance.as_args())
        elif isinstance(distance, CRSDistance):
            sw.set_distance_crs(*distance.as_args())
        else:
            raise TypeError("Distance type not supported")
        if isinstance(weight, BandwidthWeight):
            sw.set_weight_bandwidth(*weight.as_args())
        else:
            raise TypeError("Weight type not supported")
        return sw
