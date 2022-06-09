from pathlib import Path

import cv2
import numpy as np

from isegm.data.base import ISDataset
from isegm.data.sample import DSample
from isegm.utils.misc import get_labels_with_sizes, limit_longest_size


class SAICDataset(ISDataset):
    def __init__(
        self, dataset_path, images_dir_name='imgs', masks_dir_name='masks',
        max_side_size=-1, **kwargs
    ):
        super(SAICDataset, self).__init__(**kwargs)

        self.dataset_path = Path(dataset_path)
        self._images_path = self.dataset_path / images_dir_name
        self._insts_path = self.dataset_path / masks_dir_name
        self.max_side_size = max_side_size

        self.dataset_samples = [x.name for x in sorted(self._images_path.glob('*.*'))]

    def get_sample(self, index) -> DSample:
        image_name = self.dataset_samples[index]
        image_path = str(self._images_path / image_name)
        mask_path = str(self._insts_path / image_name.replace('.jpg', '.png'))

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        instances_mask = cv2.imread(mask_path)
        instances_mask = cv2.cvtColor(instances_mask, cv2.COLOR_BGR2GRAY)
        instances_mask = instances_mask.astype(np.int32)

        if self.max_side_size > 0:
            image, new_shape = limit_longest_size(image, self.max_side_size)
            instances_mask, new_shape = limit_longest_size(
                instances_mask, self.max_side_size, new_shape,
                interpolation=cv2.INTER_NEAREST
            )

        object_ids, _ = get_labels_with_sizes(instances_mask)
        return DSample(
            image, instances_mask, objects_ids=object_ids,
            sample_id=index, image_name=image_name
        )
