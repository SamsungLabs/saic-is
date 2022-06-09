import enum

import numpy as np


class IType(enum.Enum):
    contour = 'contour'
    point = 'point'
    stroke = 'stroke'


class InteractionSelector:
    def __init__(self, contours_at=(0, 1), strokes_at=(1, 3), clicks_at=(3, 1000)):
        self.clicks_at = clicks_at
        self.contours_at = contours_at
        self.strokes_at = strokes_at

    def select_itype(self, interaction_i, mask):
        size_ratio = mask.sum() / np.prod(mask.shape)
        if 0 < size_ratio < 0.1:
            return IType.point
        if self.clicks_at[0] <= interaction_i < self.clicks_at[1]:
            return IType.point
        if self.strokes_at[0] <= interaction_i < self.strokes_at[1]:
            return IType.stroke
        if self.contours_at[0] <= interaction_i < self.contours_at[1]:
            return IType.contour
        return IType.point


class RandomInteractionSelector:
    def __init__(
        self,
        itype_probs=(
            (IType.stroke, 0.3),
            (IType.contour, 0.3),
            (IType.point, 0.4)
        )
    ):
        self._itypes, self._probs = list(zip(*itype_probs))

    def select_itype(self, **kwargs):
        return np.random.choice(self._itypes, size=1, p=self._probs)
