from functools import lru_cache

import cv2
import numpy as np


@lru_cache(maxsize=16)
def get_palette(num_cls):
    palette = np.zeros(3 * num_cls, dtype=np.int32)

    for j in range(0, num_cls):
        lab = j
        i = 0

        while lab > 0:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7-i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7-i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7-i))
            i = i + 1
            lab >>= 3

    return palette.reshape((-1, 3))


def draw_probmap(x):
    return cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_HOT)


def draw_points(image, points, color, radius=3):
    image = image.copy()
    for p in points:
        if p[0] < 0:
            continue
        image = cv2.circle(image, (int(p[1]), int(p[0])), radius, color, -1)
    return image


def draw_masks(image, masks, color, alpha):
    image = image.copy()
    for mask in masks:
        color_arr = np.array(color, dtype=np.int32)
        result = color_arr[None, None, :] * mask[..., None]
        image[mask > 0] = (image[mask > 0] * (1 - alpha) + alpha * result[mask > 0]).astype(np.uint8)
    return image

 
def draw_with_blend_and_clicks(
    img, mask=None, clicks_list=None,
    alpha=0.6, pos_color=(0, 255, 0), neg_color=(255, 0, 0),
    radius=4, enlarge_last=False
):
    result = img.copy()

    if mask is not None:
        palette = get_palette(np.max(mask) + 1)
        rgb_mask = palette[mask.astype(np.uint8)]

        mask_region = (mask > 0).astype(np.uint8)
        result = result * (1 - mask_region[:, :, np.newaxis]) + \
            (1 - alpha) * mask_region[:, :, np.newaxis] * result + \
            alpha * rgb_mask
        result = result.astype(np.uint8)

    if clicks_list is not None and len(clicks_list) > 0:
        pos_points = [click.coords for click in clicks_list if click.is_positive]
        neg_points = [click.coords for click in clicks_list if not click.is_positive]
        if enlarge_last:
            if clicks_list[-1].coords in pos_points:
                last_point = pos_points.pop()
                result = draw_points(result, [last_point], pos_color, radius=int(radius * 1.5))
            else:
                last_point = neg_points.pop()
                result = draw_points(result, [last_point], neg_color, radius=int(radius * 1.5))
        result = draw_points(result, pos_points, pos_color, radius=radius)
        result = draw_points(result, neg_points, neg_color, radius=radius)

    return result


def draw_with_blend_and_lines(
    img, mask=None, interactions_list=None,
    alpha=0.6, pos_color=(0, 255, 0), neg_color=(255, 0, 0),
    thickness=10,
):
    result = img.copy()

    if mask is not None:
        palette = get_palette(np.max(mask) + 1)
        rgb_mask = palette[mask.astype(np.uint8)]

        mask_region = (mask > 0).astype(np.uint8)
        result = result * (1 - mask_region[:, :, np.newaxis]) + \
            (1 - alpha) * mask_region[:, :, np.newaxis] * result + \
            alpha * rgb_mask
        result = result.astype(np.uint8)

    if interactions_list:
        if isinstance(interactions_list[0], list):
            interactions_list = sum(interactions_list, [])
        pos_masks = [
            interaction.get_mask(interaction, img.shape[:2], thickness=thickness, filled=False)[0]
            for interaction in interactions_list if interaction.is_positive
        ]
        neg_masks = [
            interaction.get_mask(interaction, img.shape[:2], thickness=thickness, filled=False)[1]
            for interaction in interactions_list if not interaction.is_positive
        ]

        result = draw_masks(result, pos_masks, pos_color, alpha)
        result = draw_masks(result, neg_masks, neg_color, alpha)
    return result


def class_to_index(mask):
    # assert the values
    values = np.unique(mask)
    values = [v for v in values if v > 0]
    key = [-1, 0] + values
    mask_n = np.zeros_like(mask)
    for idx, k in enumerate(key):
        mask_n[mask == k] = idx - 1
    return mask_n


def draw_lines(image, points, color, radius=3):
    image = image.copy()

    if len(points) < 2:
        return image

    p0 = points[0]

    for p in points[1:]:
        if p[0] < 0:
            continue
        if len(p) == 3:
            pradius = {0: 8, 1: 6, 2: 4}[p[2]] if p[2] < 3 else 2
        else:
            pradius = max(1, radius)

        pt1 = int(p0[1]+0.5), int(p0[0]+0.5)
        pt2 = int(p[1]+0.5), int(p[0]+0.5)
        image = cv2.line(image, pt1, pt2, color, thickness=pradius)
        p0 = p

    return image


def fill_poly(image, points, color):
    image = image.copy()

    if len(points) < 2:
        return image

    pts = [np.array(points)[:, ::-1].astype(np.int32)]
    image = cv2.fillPoly(image, pts=pts, color=color)

    return image


def blend_with_interaction_mask(image, interaction, alpha=0.3, pos_color=(0, 255, 0), neg_color=(255, 0, 0)):
    image = image.copy()
    colors = [pos_color, neg_color]
    for layer_i in range(interaction.shape[2]):
        mask = np.tile(interaction[:, :, layer_i:layer_i + 1], (1, 1, 3))
        not_bg = mask > 0
        mask = np.array(colors[layer_i])[None, None, :] * mask
        image[not_bg] = (
            alpha * mask[not_bg]
            + (1 - alpha) * image[not_bg]
        )
    return image


def blend_with_mask_and_interaction(
    img, mask=None,
    alpha=0.6, palette=None,
    contours=None, interaction_mask=None, clicks=None,
    pos_color=(0, 255, 0), neg_color=(255, 0, 0),
    radius=4, fill_contours=False
):
    result = img.copy()

    if mask is not None:
        if palette is None:
            mask = class_to_index(mask)
            if mask.min() == -1:
                mask = mask.copy() + 1
                palette = get_palette(mask.max())
                palette = np.concatenate(([[255, 255, 255]], palette), axis=0)
            else:
                palette = get_palette(np.max(mask) + 1)
        rgb_mask = palette[mask.astype(np.uint8)]

        mask_region = (mask > 0).astype(np.uint8)
        result = result * (1 - mask_region[:, :, np.newaxis]) + \
            (1 - alpha) * mask_region[:, :, np.newaxis] * result + \
            alpha * rgb_mask
        result = result.astype(np.uint8)

    if contours is not None and len(contours) > 0:
        for contour in contours:
            pos_points = [coord for coord in contour.coords]
            color = pos_color if contour.is_positive else neg_color
            if fill_contours:
                result_new = fill_poly(result, pos_points, color)
            else:
                result_new = draw_lines(result, pos_points, color, radius=radius)

            result = (result * (1 - alpha) + alpha * result_new).astype(np.uint8)

    if interaction_mask is not None:
        positive = interaction_mask[:, :, 0:1]
        negative = interaction_mask[:, :, 1:2]

        if len(positive.shape) < 3 or positive.shape[2] == 1:
            positive = np.tile(positive, (1, 1, 3))
            negative = np.tile(negative, (1, 1, 3))

        neg_color_arr = np.array(neg_color, dtype=np.int32)
        pos_color_arr = np.array(pos_color, dtype=np.int32)
        neg_result = neg_color_arr[None, None, :] * negative
        pos_result = pos_color_arr[None, None, :] * positive

        result[negative > 0] = (result[negative > 0] * (1 - alpha) + alpha * neg_result[negative > 0]).astype(np.uint8)
        result[positive > 0] = (result[positive > 0] * (1 - alpha) + alpha * pos_result[positive > 0]).astype(np.uint8)

    if clicks is not None and len(clicks) > 0:
        sep = len(clicks) // 2
        pos_points = [coords for coords in clicks[:sep]]
        neg_points = [coords for coords in clicks[sep:]]
        result = draw_points(result, pos_points, pos_color, radius=radius)
        result = draw_points(result, neg_points, neg_color, radius=radius)
    return result


def save_image(output_images_path, prefix, suffix, image):
    if prefix and suffix:
        name = f'{prefix}_{suffix}.jpg'
    elif prefix:
        name = f'{prefix}.jpg'
    elif suffix:
        name = f'{suffix}.jpg'
    else:
        name = 'image.jpg'
    path = str(output_images_path / name)
    cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return path
