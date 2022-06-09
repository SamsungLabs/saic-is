from typing import Dict

from isegm.data.interaction_type import IType
from isegm.data.sample import DSample
from .base import BaseInteractionSampler


class ComposeInteractionSampler(BaseInteractionSampler):
    def __init__(self, samplers_by_itype: Dict[IType, BaseInteractionSampler], **kwargs):
        super().__init__(**kwargs)
        self.samplers_by_itype = samplers_by_itype
        self.generator = {itype: sampler.generator for itype, sampler in samplers_by_itype.items()}

    def sample_object(self, sample: DSample):
        super().sample_object(sample)
        for sampler in self.samplers_by_itype.values():
            sampler.set_object(self._selected_mask, self._selected_masks, self._neg_masks)

    def sample_interaction(self):
        interactions = {}
        for itype, sampler in self.samplers_by_itype.items():
            interactions[itype] = sampler.sample_interaction()
        return interactions
