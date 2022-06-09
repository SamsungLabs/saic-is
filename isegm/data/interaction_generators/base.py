import numpy as np

from isegm.utils.misc import mask_to_boundary, get_boundary_size


class BaseInteractionGenerator:
    def __init__(self, deterministic=False, one_component=False):
        self.deterministic = deterministic
        self.one_component = one_component

    def get_random_generator(self, mask, is_positive):
        if self.deterministic:
            seed = 1 if is_positive else 0
            seed += mask.sum()
            return np.random.RandomState(int(seed))
        else:
            return np.random


def remove_thin_border(mask, dilation_ratio=0.001):
    bsize = get_boundary_size(mask, dilation_ratio=dilation_ratio, use_mask_diag=False)
    bmask, reduced_mask = mask_to_boundary(mask.astype(np.uint8), boundary_size=bsize, with_erode=True)
    return reduced_mask
