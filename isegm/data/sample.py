import math
from copy import deepcopy

from albumentations import ReplayCompose
try:
    from albumentations.augmentations.crops.functional import crop, crop_keypoint_by_coords
    from albumentations.augmentations.geometric.functional import resize, keypoint_scale
except (ImportError, AttributeError):
    from albumentations.augmentations.functional import crop, crop_keypoint_by_coords, resize, keypoint_scale
import numpy as np

from isegm.data.contour import Contour
from isegm.data.transforms import remove_image_only_transforms
from isegm.utils.misc import get_labels_with_sizes, get_labels_with_rois


class DSample:
    def __init__(
        self, image, encoded_masks,
        objects=None, objects_ids=None, ignore_ids=None,
        sample_id=None, image_id=None, instance_id=None,
        image_name=None, mask_names=None,
        instance_classes=None,
        contours=None, contours_masks=None, synthetic_obj_ids=None,
    ):
        self.image = image
        self.sample_id = sample_id
        self.image_name = image_name
        self.instance_id = instance_id
        self.image_id = image_id

        self.set_objects(encoded_masks, objects, objects_ids, ignore_ids)
        self.set_contours(contours)
        self.set_contours_masks(contours_masks)
        self.set_mask_names(mask_names)
        self.set_instance_classes(instance_classes)
        self.set_synthetic_objects_ids(synthetic_obj_ids)

        self._augmented = False
        self._soft_mask_aug = None
        self._original_data = self.image, self._encoded_masks, deepcopy(self._objects),\
                              deepcopy(self._mask_names), deepcopy(self._instance_classes),\
                              deepcopy(self._contours), self._contours_masks, deepcopy(self._contours_masks_meta)

    def _get_inverse_mapping(self):
        inverse_index = {node['mapping']: node_id for node_id, node in self._objects.items()}
        return inverse_index

    def set_objects(self, encoded_masks, objects=None, objects_ids=None, ignore_ids=None):

        if len(encoded_masks.shape) == 2:
            encoded_masks = encoded_masks[:, :, np.newaxis]
        self._encoded_masks = encoded_masks
        self._ignored_regions = []

        if objects_ids is not None:
            if not objects_ids or not isinstance(objects_ids[0], tuple):
                assert self._encoded_masks.shape[2] == 1
                objects_ids = [(0, mask_id) for mask_id in objects_ids]

            self._objects = dict()
            for indx, obj_mapping in enumerate(objects_ids):
                self._objects[indx] = {
                    'parent': None,
                    'mapping': obj_mapping,
                    'children': []
                }

            if ignore_ids:
                if isinstance(ignore_ids[0], tuple):
                    self._ignored_regions = ignore_ids
                else:
                    self._ignored_regions = [(0, region_id) for region_id in ignore_ids]
        else:
            self._objects = deepcopy(objects)

    def set_contours(self, contours):

        if isinstance(contours, dict):
            for obj_id in contours:
                if obj_id not in self._objects:
                    raise ValueError("Invalid contours!")
            self._contours = contours
            return

        inverse_index = self._get_inverse_mapping()

        if contours is not None and len(contours) > 0:
            contours_mapping = {}
            for item in contours:
                if len(item) == 3:
                    layer_idx, mask_id, obj_contours = item
                else:
                    mask_id, obj_contours = item
                    layer_idx = 0
                obj_id = inverse_index[(layer_idx, mask_id)]
                contours_mapping[obj_id] = obj_contours
            self._contours = contours_mapping
        else:
            self._contours = None

    def set_contours_masks(self, contours_masks):

        inverse_index = self._get_inverse_mapping()

        if contours_masks is not None and len(contours_masks) > 0:
            c_masks = []
            contours_masks_meta = {}
            for c_idx, item in enumerate(contours_masks):
                if len(item) == 3:
                    layer_idx, instance_id, c_mask = item
                else:
                    instance_id, c_mask = item
                    layer_idx = 0
                key_v = (layer_idx, instance_id)
                obj_id = inverse_index[key_v]
                contours_masks_meta[obj_id] = c_idx
                c_masks.append(c_mask)
            self._contours_masks = np.stack(c_masks, axis=3)
            self._contours_masks_meta = contours_masks_meta
        else:
            self._contours_masks = None
            self._contours_masks_meta = None

    def set_mask_names(self, mask_names):

        inverse_index = self._get_inverse_mapping()

        if mask_names is not None and len(mask_names) > 0:
            mask_names_mapping = {}
            for item in mask_names:
                if len(item) == 3:
                    layer_idx, instance_id, mask_name = item
                else:
                    instance_id, mask_name = item
                    layer_idx = 0
                key_v = (layer_idx, instance_id)
                obj_id = inverse_index[key_v]
                mask_names_mapping[obj_id] = mask_name

            self._mask_names = mask_names_mapping
        else:
            self._mask_names = None

    def set_instance_classes(self, instance_classes):
        inverse_index = self._get_inverse_mapping()

        if instance_classes is not None and len(instance_classes) > 0:
            instance_classes_mapping = {}
            for item in instance_classes:
                if len(item) == 3:
                    layer_idx, instance_id, instance_class = item
                else:
                    instance_id, instance_class = item
                    layer_idx = 0
                key_v = (layer_idx, instance_id)
                obj_id = inverse_index[key_v]
                instance_classes_mapping[obj_id] = instance_class

            self._instance_classes = instance_classes_mapping
        else:
            self._instance_classes = None

    def set_synthetic_objects_ids(self, synthetic_obj_ids):
        synthetic_obj_ids = [] if synthetic_obj_ids is None else synthetic_obj_ids
        synthetic_obj_ids = set(synthetic_obj_ids)
        for obj_id in self._objects:
            self._objects[obj_id]['synthetic'] = obj_id in synthetic_obj_ids

    def set_main_obj(self, obj_id):
        for node in self._objects.values():
            node['is_main'] = False
        self._objects[obj_id]['is_main'] = True

    def get_main_object_id(self):
        main_obj_id = [obj_id for obj_id, obj_info in self._objects.items() if 'is_main' in obj_info and obj_info['is_main']]
        if len(main_obj_id) != 1:
            return None
        return main_obj_id[0]

    @staticmethod
    def keypoints_from_contours(contours_mapping):
        keypoints = []
        obj_ids = []
        contour_ids = []
        kp2contours_mapping = []
        for obj_id, contours in contours_mapping.items():
            kp2contours_obj_mapping = []
            for contour_idx, contour in enumerate(contours):
                coords = contour.coords
                keypoints.extend(coords)
                contour_id = (obj_id, contour_idx)
                obj_ids.extend([obj_id] * len(coords))
                contour_ids.extend([contour_id] * len(coords))
                kp2contours_obj_mapping.append((contour.is_positive, len(coords)))
            kp2contours_mapping.append((obj_id, kp2contours_obj_mapping))
        return keypoints, obj_ids, contour_ids, kp2contours_mapping

    @staticmethod
    def contours_from_keypoints(keypoints, contours_mapping):
        contours_mapping_new = dict()
        start_idx = 0
        for obj_id, contours_info in contours_mapping:
            contours_new = []
            for is_positive, coord_len in contours_info:
                coords = [kp for kp in keypoints[start_idx:start_idx + coord_len] if None not in kp]
                start_idx = start_idx + coord_len
                contour_new = Contour(is_positive, coords=coords)
                contours_new.append(contour_new)
            contours_mapping_new[obj_id] = contours_new
        return contours_mapping_new

    def augment(self, augmentator, min_area=1, max_obj_roi=1.0):
        self.reset_augmentation()
        keypoints = []
        kp_contour_ids = []
        kp2contours_mapping = []
        if self.has_contours():
            keypoints, _, kp_contour_ids, kp2contours_mapping = self.keypoints_from_contours(self._contours)
        to_augment_mask = self._encoded_masks

        contours_masks = self._contours_masks
        no_contour_masks = False
        if contours_masks is not None:
            cm_shape = contours_masks.shape
            contours_masks = contours_masks.reshape((cm_shape[0], cm_shape[1], cm_shape[2]*cm_shape[3]))
        else:
            contours_masks = np.zeros(self.image.shape[:2] + (2,), dtype=np.int32)
            no_contour_masks = True
        aug_output = augmentator(image=self.image, mask=to_augment_mask, objects_info=self._objects,
                                 keypoints=keypoints, keypoints_contour_ids=kp_contour_ids,
                                 contours_masks=contours_masks)
        self.image = aug_output['image']
        self._encoded_masks = aug_output['mask']

        contours = None
        if 'keypoints' in aug_output and self.has_contours():
            contours = self.contours_from_keypoints(aug_output['keypoints'], kp2contours_mapping)
        self._contours = contours
        cmasks = aug_output['contours_masks'] if 'contours_masks' in aug_output else None
        cmasks = None if no_contour_masks else cmasks
        self._contours_masks = cmasks.reshape(cmasks.shape[:2] + (2, -1)) if cmasks is not None else None

        aug_replay = aug_output.get('replay', None)
        if aug_replay:
            assert len(self._ignored_regions) == 0
            mask_replay = remove_image_only_transforms(aug_replay)
            self._soft_mask_aug = ReplayCompose._restore_for_replay(mask_replay)

        self._compute_objects_areas()
        self.remove_small_objects(min_area=min_area)

        # self._compute_objects_rois()
        # self.remove_too_big_objects(max_object_frame_iou=max_obj_roi)

        self._augmented = True

    def reset_augmentation(self):
        if not self._augmented:
            return
        orig_image, orig_masks, orig_objects, \
        orig_mask_names, orig_instance_classes, \
        orig_contours, orig_contours_masks, orig_contours_masks_meta = self._original_data

        self.image = orig_image
        self._encoded_masks = orig_masks
        self._objects = deepcopy(orig_objects)
        self._mask_names = deepcopy(orig_mask_names)
        self._instance_classes = deepcopy(orig_instance_classes)
        self._contours = deepcopy(orig_contours)
        self._contours_masks = orig_contours_masks
        self._contours_masks_meta = deepcopy(orig_contours_masks_meta)

        self._augmented = False
        self._soft_mask_aug = None

    def remove_small_objects(self, min_area):
        if self._objects and not 'area' in list(self._objects.values())[0]:
            self._compute_objects_areas()

        for obj_id, obj_info in list(self._objects.items()):
            if obj_info['area'] < min_area:
                self._remove_object(obj_id)

    def remove_too_big_objects(self, max_object_frame_iou):
        if self._objects and not 'roi' in list(self._objects.values())[0]:
            self._compute_objects_rois()

        for obj_id, obj_info in list(self._objects.items()):
            bbox = obj_info['roi']
            h, w = self.shape
            box_area = (bbox[1] - bbox[0]) * (bbox[3] - bbox[2])
            iou = float(box_area) / (w * h)
            if iou > max_object_frame_iou or math.isclose(box_area, 0.0):
                self._remove_object(obj_id)

    def get_object_mask(self, obj_id):
        layer_indx, mask_id = self._objects[obj_id]['mapping']
        obj_mask = (self._encoded_masks[:, :, layer_indx] == mask_id).astype(np.int32)
        if self._ignored_regions:
            for layer_indx, mask_id in self._ignored_regions:
                ignore_mask = self._encoded_masks[:, :, layer_indx] == mask_id
                obj_mask[ignore_mask] = -1

        return obj_mask

    def get_object_class(self, obj_id):
        return self._instance_classes[obj_id]

    def get_object_contours_mask(self, obj_id):
        if not self.has_contours_masks():
            return None
        layer_index = self._contours_masks_meta[obj_id]
        result = self._contours_masks[:,:,:,layer_index].astype(np.int32)
        return result

    def get_object_contours(self, obj_id):
        if not self.has_contours():
            return []
        obj_contours = self._contours[obj_id]
        return obj_contours

    def get_encoded_masks(self):
        encoded_mask = self._encoded_masks.copy()
        if self._ignored_regions:
            for layer_indx, mask_id in self._ignored_regions:
                layer_mask = encoded_mask[:, :, layer_indx]
                ignore_mask = layer_mask == mask_id
                layer_mask[ignore_mask] = -1
        return encoded_mask

    def get_soft_object_mask(self, obj_id):
        assert self._soft_mask_aug is not None
        original_encoded_masks = self._original_data[1]
        layer_indx, mask_id = self._objects[obj_id]['mapping']
        obj_mask = (original_encoded_masks[:, :, layer_indx] == mask_id).astype(np.float32)
        obj_mask = self._soft_mask_aug(image=obj_mask, mask=original_encoded_masks)['image']
        return np.clip(obj_mask, 0, 1)

    def get_background_mask(self):
        return np.max(self._encoded_masks, axis=2) == 0

    def get_mask_name(self, obj_id):
        if self._mask_names is None:
            return None
        return self._mask_names[obj_id]

    def has_synthetic_objects(self):
        values = [self._objects[obj_id].get('synthetic', False) for obj_id in self._objects]
        return np.any(values)

    def has_contours(self):
        return self._contours is not None and any([contour is not None for contour in self._contours.values()])

    def has_contours_masks(self):
        return self._contours_masks is not None

    @property
    def shape(self):
        return self._encoded_masks.shape[:2]

    @property
    def objects_ids(self):
        return list(self._objects.keys())

    @property
    def synthetic_objects_ids(self):
        return [obj_id for obj_id in self._objects if self._objects[obj_id]['synthetic']]

    @property
    def gt_masks(self):
        return [self.get_object_mask(obj_id) for obj_id in self.objects_ids]

    @property
    def gt_mask(self):
        assert len(self._objects) == 1
        return self.get_object_mask(self.objects_ids[0])

    @property
    def gt_mask_name(self):
        assert len(self._objects) == 1
        return self.get_mask_name(self.objects_ids[0])

    @property
    def gt_contours_mask(self):
        assert len(self._objects) == 1
        return self.get_object_contours_mask(self.objects_ids[0])

    @property
    def gt_contours(self):
        assert len(self._objects) == 1
        obj_contours = self.get_object_contours(self.objects_ids[0])
        return obj_contours

    @property
    def root_objects(self):
        return [obj_id for obj_id, obj_info in self._objects.items() if obj_info['parent'] is None]

    @property
    def contours(self):
        return self._contours

    def _compute_objects_areas(self):
        inverse_index = self._get_inverse_mapping()
        ignored_regions_keys = set(self._ignored_regions)

        for layer_indx in range(self._encoded_masks.shape[2]):
            objects_ids, objects_areas = get_labels_with_sizes(self._encoded_masks[:, :, layer_indx])
            for obj_id, obj_area in zip(objects_ids, objects_areas):
                inv_key = (layer_indx, obj_id)
                if inv_key in ignored_regions_keys:
                    continue
                try:
                    self._objects[inverse_index[inv_key]]['area'] = obj_area
                    del inverse_index[inv_key]
                except KeyError:
                    layer = self._encoded_masks[:, :, layer_indx]
                    layer[layer == obj_id] = 0
                    self._encoded_masks[:, :, layer_indx] = layer

        for obj_id in inverse_index.values():
            self._objects[obj_id]['area'] = 0

    def _compute_objects_rois(self):
        inverse_index = self._get_inverse_mapping()
        ignored_regions_keys = set(self._ignored_regions)

        for layer_indx in range(self._encoded_masks.shape[2]):
            objects_ids, objects_rois = get_labels_with_rois(self._encoded_masks[:, :, layer_indx])
            for obj_id, obj_roi in zip(objects_ids, objects_rois):
                inv_key = (layer_indx, obj_id)
                if inv_key in ignored_regions_keys:
                    continue
                try:
                    self._objects[inverse_index[inv_key]]['roi'] = obj_roi
                    del inverse_index[inv_key]
                except KeyError:
                    layer = self._encoded_masks[:, :, layer_indx]
                    layer[layer == obj_id] = 0
                    self._encoded_masks[:, :, layer_indx] = layer

        for obj_id in inverse_index.values():
            self._objects[obj_id]['roi'] = (0, 0, 0, 0)

    def _remove_object(self, obj_id):
        obj_info = self._objects[obj_id]
        obj_parent = obj_info['parent']
        for child_id in obj_info['children']:
            self._objects[child_id]['parent'] = obj_parent

        if obj_parent is not None:
            parent_children = self._objects[obj_parent]['children']
            parent_children = [x for x in parent_children if x != obj_id]
            self._objects[obj_parent]['children'] = parent_children + obj_info['children']

        if self.has_contours():
            if obj_id in self._contours:
                self._contours.pop(obj_id)

        if self._mask_names is not None:
            if obj_id in self._mask_names:
                self._mask_names.pop(obj_id)

        if self._instance_classes is not None:
            if obj_id in self._instance_classes:
                self._instance_classes.pop(obj_id)

        if self.has_contours_masks():
            if len(self._contours_masks_meta) == 1 and obj_id in self._contours_masks_meta:
                self._contours_masks = None
                self._contours_masks_meta = None
            else:
                layer_index = self._contours_masks_meta[obj_id]
                self._contours_masks = np.delete(self._contours_masks, layer_index, 3)
                contours_masks_meta = {}
                for k, v in self._contours_masks_meta.items():
                    if k == obj_id:
                        continue
                    if v >= layer_index:
                        v -= 1
                    contours_masks_meta[k] = v
                self._contours_masks_meta = contours_masks_meta

        del self._objects[obj_id]

    def __len__(self):
        return len(self._objects)

    @property
    def objects(self):
        return self._objects
