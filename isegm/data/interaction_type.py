import enum
from itertools import product


import numpy as np


class IType(enum.Enum):
    contour = 'contour'
    point = 'point'
    stroke = 'stroke'


class BaseInteractionSelector:
    def __init__(self):
        self.itypes = []

    def select_itype(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def name(self):
        return '_'.join([t.name for t in self.itypes])


class SingleInteractionSelector(BaseInteractionSelector):
    def __init__(self, input_type):
        super().__init__()
        self.itypes = [input_type]
        self.input_type = input_type

    def select_itype(self, *args, **kwargs):
        return self.input_type


class InteractionSelector(BaseInteractionSelector):
    def __init__(
        self,
        itype_iters=(
            (IType.contour, (0, 1)),
            (IType.stroke, (1, 3)),
            (IType.point, (3, 1000))
        )
    ):
        super().__init__()
        self.itype_iters = dict(itype_iters)
        self.itypes = list(zip(*itype_iters))[0]

    def select_itype(self, interaction_i, mask):
        size_ratio = mask.sum() / np.prod(mask.shape)
        if 0 < size_ratio < 0.1:
            return IType.point
        for itype, bounds in self.itype_iters.items():
            max_bound = bounds[1] if bounds[1] is not None else interaction_i + 1
            if bounds[0] <= interaction_i < max_bound:
                return itype
        return IType.point


class RandomInteractionSelector(BaseInteractionSelector):
    def __init__(
        self,
        itype_probs=(
            (IType.stroke, 0.33),
            (IType.contour, 0.33),
            (IType.point, 0.34)
        )
    ):
        super().__init__()
        self.itypes, self._probs = list(zip(*itype_probs))

    def select_itype(self, *args, **kwargs):
        if len(self.itypes) == 1:
            return self.itypes[0]
        return np.random.choice(self.itypes, size=1, p=self._probs)[0]


class ProductInteractionSelector(BaseInteractionSelector):
    def __init__(self):
        super().__init__()
        self.itypes = [IType.contour, IType.stroke, IType.point]
        self.product_index = 0
        self.possible_products = list(product(range(len(self.itypes)), repeat=len(self.itypes)))

    def select_itype(self, interaction_i, *args, **kwargs):
        itype_indices = self.possible_products[self.product_index]
        if interaction_i >= len(itype_indices):
            return IType.point
        itype_i = itype_indices[interaction_i]
        return self.itypes[itype_i]

    def next_product(self):
        self.product_index += 1

    def reset(self):
        self.product_index = 0

    def __len__(self):
        return len(self.possible_products)
