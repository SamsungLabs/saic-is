from collections import defaultdict

import cv2
import numpy as np


def get_masks_hierarchy(masks, masks_meta):
    order = sorted(list(enumerate(masks_meta)), key=lambda x: x[1][1])[::-1]
    hierarchy = defaultdict(list)

    def check_inter(i, j):
        assert masks_meta[i][1] >= masks_meta[j][1]
        bbox_i, bbox_j = masks_meta[i][0], masks_meta[j][0]
        bbox_score = get_bbox_intersection(bbox_i, bbox_j) / get_bbox_area(bbox_j)
        if bbox_score < 0.7:
            return False

        mask_i, mask_j = masks[i], masks[j]
        mask_score = np.logical_and(mask_i, mask_j).sum() / masks_meta[j][1]
        return mask_score > 0.8

    def get_root_indx(root_indx, check_indx):
        children = hierarchy[root_indx]
        for child_indx in children:
            if masks_meta[child_indx][1] < masks_meta[check_indx][1]:
                continue
            result_indx = get_root_indx(child_indx, check_indx)
            if result_indx is not None:
                return result_indx

        if check_inter(root_indx, check_indx):
            return root_indx

        return None

    used_masks = np.zeros(len(masks), dtype=np.bool)
    parents = [None] * len(masks)
    node_level = [0] * len(masks)
    for ti in range(len(masks) - 1):
        for tj in range(ti + 1, len(masks)):
            i = order[ti][0]
            j = order[tj][0]

            assert i != j
            if used_masks[j] or not check_inter(i, j):
                continue

            ni = get_root_indx(i, j)
            assert ni != j and parents[j] is None
            hierarchy[ni].append(j)
            used_masks[j] = True
            parents[j] = ni
            node_level[j] = node_level[ni] + 1

    hierarchy = [hierarchy[i] for i in range(len(masks))]
    hierarchy = {i: {'children': hierarchy[i],
                     'parent': parents[i],
                     'node_level': node_level[i]
                     }
                 for i in range(len(masks))}
    return hierarchy


def get_bbox_intersection(b1, b2):
    h_i = get_segments_intersection(b1[:2], b2[:2])
    w_i = get_segments_intersection(b1[2:4], b2[2:4])
    return h_i * w_i


def get_segments_intersection(s1, s2):
    a, b = s1
    c, d = s2
    return max(0, min(b, d) - max(a, c) + 1)


def get_bbox_area(bbox):
    return (bbox[1] - bbox[0] + 1) * (bbox[3] - bbox[2] + 1)


def get_iou(mask1, mask2):
    intersection_area = np.logical_and(mask1, mask2).sum()
    union_area = np.logical_or(mask1, mask2).sum()
    return intersection_area / union_area


def encode_masks(masks):
    layers = [np.zeros(masks[0].shape, dtype=np.uint8)]
    layers_objs = [[]]
    objs_mapping = [(None, None)] * len(masks)
    ordered_masks = sorted(list(enumerate(masks)), key=lambda x: x[1].sum())[::-1]
    for global_id, obj_mask in ordered_masks:
        for layer_indx, (layer_mask, layer_objs) in enumerate(zip(layers, layers_objs)):
            if len(layer_objs) >= 255:
                continue
            if np.all(layer_mask[obj_mask] == 0):
                layer_objs.append(global_id)
                local_id = len(layer_objs)
                layer_mask[obj_mask] = local_id
                objs_mapping[global_id] = (layer_indx, local_id)
                break
        else:
            new_layer = np.zeros_like(layers[-1])
            new_layer[obj_mask] = 1
            objs_mapping[global_id] = (len(layers), 1)
            layers.append(new_layer)
            layers_objs.append([global_id])

    layers = [cv2.imencode('.png', x)[1] for x in layers]
    return layers, objs_mapping
