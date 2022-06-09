import math
import random

import cv2
import numpy as np

from isegm.data.sample import DSample


class BaseInteractionSampler:
    def __init__(
        self,
        max_objects,
        expand_ratio=0.1, expand_mask_relative=True,
        positive_erode_prob=0.9, positive_erode_iters=3,
        negative_bg_prob=0.1, negative_other_prob=0.4, negative_border_prob=0.5,
        merge_objects_prob=0.0, max_num_merged_objects=2,
        use_hierarchy=False, soft_targets=False,
        neg_prob=0.5,
    ):
        self.max_objects = max_objects
        self.expand_ratio = expand_ratio
        self.expand_mask_relative = expand_mask_relative
        self.positive_erode_prob = positive_erode_prob
        self.positive_erode_iters = positive_erode_iters
        self.merge_objects_prob = merge_objects_prob
        self.use_hierarchy = use_hierarchy
        self.soft_targets = soft_targets
        self._selected_mask = None
        self._selected_masks = None
        self.generator = None
        self.neg_prob = neg_prob

        if max_num_merged_objects == -1:
            max_num_merged_objects = max_objects
        self.max_num_merged_objects = max_num_merged_objects

        self.neg_strategies = ['bg', 'other', 'border']
        self.neg_strategies_prob = [negative_bg_prob, negative_other_prob, negative_border_prob]
        assert math.isclose(sum(self.neg_strategies_prob), 1.0)

        self._neg_masks = None
        self._selected_mask = None
        self._selected_masks = None

    def sample_object(self, sample: DSample):
        if len(sample) == 0:
            bg_mask = sample.get_background_mask()
            self.selected_mask = np.zeros_like(bg_mask, dtype=np.float32)
            self._selected_masks = [[]]
            self._neg_masks = {strategy: bg_mask for strategy in self.neg_strategies}
            self._neg_masks['required'] = []
            return

        gt_mask, pos_masks, neg_masks = self._sample_mask(sample)
        binary_gt_mask = gt_mask > 0.5 if self.soft_targets else gt_mask > 0

        self.selected_mask = gt_mask
        self._selected_masks = pos_masks

        neg_mask_bg = np.logical_not(binary_gt_mask)
        neg_mask_border = self._get_border_mask(binary_gt_mask)
        if len(sample) <= len(self._selected_masks):
            neg_mask_other = neg_mask_bg
        else:
            neg_mask_other = np.logical_and(
                np.logical_not(sample.get_background_mask()), np.logical_not(binary_gt_mask)
            )

        self._neg_masks = {
            'bg': neg_mask_bg,
            'other': neg_mask_other,
            'border': neg_mask_border,
            'required': neg_masks
        }

    def _sample_mask(self, sample: DSample):
        root_obj_ids = sample.root_objects

        if len(root_obj_ids) > 1 and random.random() < self.merge_objects_prob:
            max_selected_objects = min(len(root_obj_ids), self.max_num_merged_objects)
            num_selected_objects = np.random.randint(2, max_selected_objects + 1)
            random_ids = random.sample(root_obj_ids, num_selected_objects)
        else:
            random_ids = [random.choice(root_obj_ids)]

        gt_mask = None
        pos_segments = []
        neg_segments = []
        for obj_id in random_ids:
            obj_gt_mask, obj_pos_segments, obj_neg_segments = self._sample_from_masks_layer(obj_id, sample)
            if gt_mask is None:
                gt_mask = obj_gt_mask
            else:
                gt_mask = np.maximum(gt_mask, obj_gt_mask)

            pos_segments.extend(obj_pos_segments)
            neg_segments.extend(obj_neg_segments)
        pos_masks = [self._positive_erode(x) for x in pos_segments]
        neg_masks = [self._positive_erode(x) for x in neg_segments]

        return gt_mask, pos_masks, neg_masks

    def _sample_from_masks_layer(self, obj_id, sample: DSample):
        objs_tree = sample.objects

        if not self.use_hierarchy:
            node_mask = sample.get_object_mask(obj_id)
            gt_mask = sample.get_soft_object_mask(obj_id) if self.soft_targets else node_mask
            return gt_mask, [node_mask], []

        def _select_node(node_id):
            info = objs_tree[node_id]
            if not info['children'] or random.random() < 0.5:
                return node_id
            return _select_node(random.choice(info['children']))

        selected_node = _select_node(obj_id)
        node_info = objs_tree[selected_node]
        node_mask = sample.get_object_mask(selected_node)
        gt_mask = sample.get_soft_object_mask(selected_node) if self.soft_targets else node_mask
        pos_mask = node_mask.copy()

        negative_segments = []
        if node_info['parent'] is not None and node_info['parent'] in objs_tree:
            parent_mask = sample.get_object_mask(node_info['parent'])
            negative_segments.append(np.logical_and(parent_mask, np.logical_not(node_mask)))

        for child_id in node_info['children']:
            if objs_tree[child_id]['area'] / node_info['area'] < 0.10:
                child_mask = sample.get_object_mask(child_id)
                pos_mask = np.logical_and(pos_mask, np.logical_not(child_mask))

        if node_info['children']:
            max_disabled_children = min(len(node_info['children']), 3)
            num_disabled_children = np.random.randint(0, max_disabled_children + 1)
            disabled_children = random.sample(node_info['children'], num_disabled_children)

            for child_id in disabled_children:
                child_mask = sample.get_object_mask(child_id)
                pos_mask = np.logical_and(pos_mask, np.logical_not(child_mask))
                if self.soft_targets:
                    soft_child_mask = sample.get_soft_object_mask(child_id)
                    gt_mask = np.minimum(gt_mask, 1.0 - soft_child_mask)
                else:
                    gt_mask = np.logical_and(gt_mask, np.logical_not(child_mask))
                negative_segments.append(child_mask)

        return gt_mask, [pos_mask], negative_segments

    def _positive_erode(self, mask):
        if random.random() > self.positive_erode_prob:
            return mask

        kernel = np.ones((3, 3), np.uint8)
        eroded_mask = cv2.erode(mask.astype(np.uint8),
                                kernel, iterations=self.positive_erode_iters).astype(np.bool)

        if eroded_mask.sum() > 10:
            return eroded_mask
        else:
            return mask

    def _get_border_mask(self, mask):
        if self.expand_mask_relative:
            expand_r = int(np.ceil(self.expand_ratio * np.sqrt(mask.sum())))
        else:
            expand_r = int(np.ceil(self.expand_ratio * np.sqrt(
                mask.shape[0] ** 2 + mask.shape[1] ** 2
            )))
        kernel = np.ones((3, 3), np.uint8)
        expanded_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=expand_r)
        expanded_mask[mask.astype(np.bool)] = 0
        return expanded_mask

    def sample_interaction(self):
        raise NotImplementedError

    def set_object(self, selected_mask, selected_masks, neg_masks):
        self._selected_mask = selected_mask
        self._selected_masks = selected_masks
        self._neg_masks = neg_masks

    @property
    def selected_mask(self):
        assert self._selected_mask is not None
        return self._selected_mask

    @selected_mask.setter
    def selected_mask(self, mask):
        self._selected_mask = mask[np.newaxis, :].astype(np.float32)
