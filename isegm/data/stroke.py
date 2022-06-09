from isegm.data.contour import Contour


class Stroke(Contour):
    def __init__(self, is_positive, coords=None):
        super().__init__(is_positive, coords)