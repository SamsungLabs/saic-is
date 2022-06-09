import pickle
import random

import numpy as np
import torch
from torchvision import transforms

from isegm.data.interaction_samplers.points_sampler import MultiPointSampler
from isegm.data.interaction_type import IType
from isegm.data.sample import DSample


class ISDataset(torch.utils.data.dataset.Dataset):
    def __init__(
        self,
        augmentator=None,
        interactive_info_sampler=MultiPointSampler(max_objects=12),
        min_object_area=0, keep_background_prob=0.0,
        with_image_info=False,
        samples_scores_path=None, samples_scores_gamma=1.0,
        epoch_len=-1,
        max_object_frame_iou=0.97,
        input_type=IType.point
    ):
        super(ISDataset, self).__init__()
        self.epoch_len = epoch_len
        self.augmentator = augmentator
        self.min_object_area = min_object_area
        self.keep_background_prob = keep_background_prob
        self.interactive_info_sampler = interactive_info_sampler
        self.with_image_info = with_image_info
        self.samples_precomputed_scores = self._load_samples_scores(samples_scores_path, samples_scores_gamma)
        self.to_tensor = transforms.ToTensor()
        self.max_object_frame_iou = max_object_frame_iou
        self.input_type = input_type

        self.dataset_samples = None

    def __getitem__(self, index):
        if self.samples_precomputed_scores is not None:
            index = np.random.choice(self.samples_precomputed_scores['indices'],
                                     p=self.samples_precomputed_scores['probs'])
        else:
            if self.epoch_len > 0:
                index = random.randrange(0, len(self.dataset_samples))
        sample = self.get_sample(index)

        sample = self.augment_sample(sample)
        self.interactive_info_sampler.sample_object(sample)

        interactive_info = self.interactive_info_sampler.sample_interaction()
        interactive_info = {
            f'{itype.name}_interactive_info': torch.Tensor(np.array(info).astype(np.float32))
            for itype, info in interactive_info.items()
        }
        mask = self.interactive_info_sampler.selected_mask

        output = {
            'images': sample.image,
            'instances': mask,
            **interactive_info
        }

        output['images'] = self.to_tensor(output['images'])
        output['instances'] = torch.Tensor(output['instances'])

        if self.with_image_info:
            output['image_info'] = {}
            output['image_info']['sample_id'] = sample.sample_id
            output['image_info']['imname'] = sample.image_name
        return output

    def augment_sample(self, sample) -> DSample:
        if self.augmentator is None:
            return sample

        valid_augmentation = False
        max_try = 3
        try_index = 0
        while not valid_augmentation:
            sample.augment(self.augmentator, min_area=self.min_object_area, max_obj_roi=self.max_object_frame_iou)
            keep_sample = (
                self.keep_background_prob < 0.0
                or random.random() < self.keep_background_prob
                or len(sample) == 0
            )
            valid_augmentation = len(sample) > 0 or keep_sample
            try_index += 1
            if try_index >= max_try:
                break

        return sample

    def get_sample(self, index) -> DSample:
        raise NotImplementedError

    def __len__(self):
        if self.epoch_len > 0:
            return self.epoch_len
        else:
            return self.get_samples_number()

    def get_samples_number(self):
        return len(self.dataset_samples)

    @staticmethod
    def _load_samples_scores(samples_scores_path, samples_scores_gamma):
        if samples_scores_path is None:
            return None

        with open(samples_scores_path, 'rb') as f:
            images_scores = pickle.load(f)

        probs = np.array([(1.0 - x[2]) ** samples_scores_gamma for x in images_scores])
        probs /= probs.sum()
        samples_scores = {
            'indices': [x[0] for x in images_scores],
            'probs': probs
        }
        print(f'Loaded {len(probs)} weights with gamma={samples_scores_gamma}')
        return samples_scores

    @property
    def name(self):
        return ''
