from __future__ import print_function, absolute_import

from .SoftmaxNeigLoss import SoftmaxNeigLoss
from .KNNSoftmax import KNNSoftmax
from .NeighbourLoss import NeighbourLoss
from .triplet import TripletLoss
from .triplet_no_hard_mining import TripletLossNoHardMining
from .CenterTriplet import CenterTripletLoss
from .GaussianMetric import GaussianMetricLoss
from .HistogramLoss import HistogramLoss
from .BatchAll import BatchAllLoss
from .NeighbourLoss import NeighbourLoss
from .DistanceMatchLoss import DistanceMatchLoss
from .NeighbourHardLoss import NeighbourHardLoss
from .DistWeightLoss import DistWeightLoss
from .BinDevianceLoss import BinDevianceLoss
from .BinBranchLoss import BinBranchLoss
from .MarginDevianceLoss import MarginDevianceLoss
from .MarginPositiveLoss import MarginPositiveLoss
from .ContrastiveLoss import ContrastiveLoss
from .DistWeightContrastiveLoss import DistWeightContrastiveLoss
from .DistWeightDevianceLoss import DistWeightBinDevianceLoss
from .DistWeightDevBranchLoss import DistWeightDevBranchLoss
from .DistWeightNeighbourLoss import DistWeightNeighbourLoss
from .BDWNeighbourLoss import BDWNeighbourLoss
from .EnsembleDWNeighbourLoss import EnsembleDWNeighbourLoss
from .BranchKNNSoftmax import BranchKNNSoftmax
from .LiftedStructure import LiftedStructureLoss
from .ms_loss import MultiSimilarityLoss
from .angular import AngularLoss, NPairAngularLoss, NPairLoss


__factory = {
    "softneig": SoftmaxNeigLoss,
    "knnsoftmax": KNNSoftmax,
    "neighbour": NeighbourLoss,
    "triplet": TripletLoss,
    "triplet_no_hard_mining": TripletLossNoHardMining,
    "histogram": HistogramLoss,
    "gaussian": GaussianMetricLoss,
    "batchall": BatchAllLoss,
    "neighard": NeighbourHardLoss,
    "bin": BinDevianceLoss,
    "binbranch": BinBranchLoss,
    "margin": MarginDevianceLoss,
    "positive": MarginPositiveLoss,
    "con": ContrastiveLoss,
    "distweight": DistWeightLoss,
    "distance_match": DistanceMatchLoss,
    "dwcon": DistWeightContrastiveLoss,
    "dwdev": DistWeightBinDevianceLoss,
    "dwneig": DistWeightNeighbourLoss,
    "dwdevbranch": DistWeightDevBranchLoss,
    "bdwneig": BDWNeighbourLoss,
    "edwneig": EnsembleDWNeighbourLoss,
    "branchKS": BranchKNNSoftmax,
    "LiftedStructure": LiftedStructureLoss,
    "Angular": AngularLoss,
    "NPairAngular": NPairAngularLoss,
    "MSLoss": MultiSimilarityLoss,
}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    """
    Create a loss instance.

    Parameters
    ----------
    name : str
        the name of loss function
    """
    if name not in __factory:
        raise KeyError("Unknown loss:", name)
    return __factory[name](*args, **kwargs)
