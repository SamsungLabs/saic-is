import torch
from functools import partial
from easydict import EasyDict as edict
from albumentations import *

from isegm.data.datasets import *
from isegm.model.losses import *
from isegm.data.transforms import *
from isegm.engine.trainer import ISTrainer
from isegm.model.metrics import AdaptiveIoU, Accuracy
from isegm.data.interaction_samplers import (
    ComposeInteractionSampler, MultiPointSampler, MultiContourSampler, MultiStrokesSampler
)
from isegm.data.interaction_type import IType, InteractionSelector, RandomInteractionSelector, SingleInteractionSelector
from isegm.data.interaction_generators import ContourGenerator, PointGenerator, StrokeGenerator, AxisTransformType
from isegm.utils.log import logger
from isegm.model import initializer

from isegm.model.is_hrnet_model import HRNetModel
from isegm.model.is_segformer_model import SegFormerModel
