from isegm.inference.transforms import ZoomIn
from .base import BasePredictor


def get_predictor(
    net, device,
    with_flip=True,
    fixed_size=None,
    zoom_in_params=dict(),
    predictor_params=None
):
    predictor_params_ = {
        'optimize_after_n_clicks': 1,
        'fixed_size': fixed_size
    }

    if zoom_in_params is not None:
        zoom_in = ZoomIn(**zoom_in_params)
    else:
        zoom_in = None

    if predictor_params is not None:
        predictor_params_.update(predictor_params)
    predictor = BasePredictor(net, device, zoom_in=zoom_in, with_flip=with_flip, **predictor_params_)

    return predictor
