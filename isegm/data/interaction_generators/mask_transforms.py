import cv2
try:
    from albumentations.augmentations.geometric.functional import elastic_transform
except (ImportError, AttributeError):
    from albumentations.augmentations.functional import elastic_transform
import numpy as np

from isegm.utils.misc import bounded_randint, get_bbox_from_mask, expand_bbox, clamp_bbox


MIN_FILTER_SIZE = 1
MIN_SIZE_RATIO = 64
MAX_SIZE_RATIO = 8
FILTER_TYPES = ('median', 'gaussian')


def get_working_area(mask, padding, shrink_to_object=True):
    """work only on the area where the mask is not zero, drastically reducing the computation time"""
    h, w = mask.shape[0], mask.shape[1]

    if shrink_to_object:
        bbox = get_bbox_from_mask(mask)
        bbox = expand_bbox(bbox, 1.1)
    else:
        bbox = 0, mask.shape[0], 0, mask.shape[1]
    rmin, rmax, cmin, cmax = bbox
    # add padding
    bbox = rmin - padding, rmax + padding, cmin - padding, cmax + padding
    bbox = clamp_bbox(bbox, -padding, h - 1 + padding, -padding, w - 1 + padding)

    rmin, rmax, cmin, cmax = bbox
    pad_h0 = max(0, -rmin)
    pad_h1 = max(0, -(h - 1 - rmax))
    pad_w0 = max(0, -cmin)
    pad_w1 = max(0, -(w - 1 - cmax))

    h1, w1 = rmax - rmin + 1, cmax - cmin + 1
    mask_pad = np.zeros((h1, w1), np.uint8)

    rmin = max(0, rmin)
    cmin = max(0, cmin)
    rmax = min(rmax, h - 1)
    cmax = min(cmax, w - 1)

    assert rmin <= h - 1
    assert cmin <= w - 1
    assert rmax >= 0
    assert cmax >= 0

    mask_pad[pad_h0:h1 - pad_h1, pad_w0:w1 - pad_w1] = mask[rmin:rmax + 1, cmin:cmax + 1]

    bbox = rmin, rmax, cmin, cmax
    pad_values = pad_h0, pad_h1, pad_w0, pad_w1
    metadata = bbox, pad_values

    return mask_pad, metadata


def put_result_back(mask_orig, result, metadata, inplace=False):
    if not inplace:
        mask_orig = np.array(mask_orig)
    bbox, padding = metadata
    rmin, rmax, cmin, cmax = bbox
    pad_h0, pad_h1, pad_w0, pad_w1 = padding
    h1, w1 = result.shape[:2]
    mask_orig[rmin:rmax + 1, cmin:cmax + 1] = result[pad_h0:h1 - pad_h1, pad_w0:w1 - pad_w1]
    return mask_orig


def remove_holes(mask):
    mask = mask.astype(np.uint8)

    h, w = mask.shape[:2]
    result_pad = np.zeros((h + 2, w + 2), np.uint8)
    result_pad[1:-1, 1:-1] = mask
    fill_mask = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(result_pad, fill_mask, (0, 0), 2)
    result_pad[result_pad != 2] = 1
    result_pad[result_pad == 2] = 0
    result = result_pad[1:-1, 1:-1]

    return result


def random_dilate_or_erode(
    mask, generator=np.random, dilate_scale_range=(64., 128.),
    erode_scale_range=(64., 128.), min_erode_ratio=0.3, shrink_to_object=True,
    dilate_prob=0.5,
):

    mask = mask.astype(np.uint8)
    obj_bbox = get_bbox_from_mask(mask)

    work_area, metadata = get_working_area(mask, 2, shrink_to_object)
    orig_area = np.count_nonzero(work_area)

    def _get_params(erode=None):
        prob = dilate_prob
        if erode is not None:
            prob = 0.0 if erode else 1.0
        if generator.uniform(0., 1.) < prob:
            main_func = cv2.dilate
            scale = generator.uniform(*dilate_scale_range)
            iterations = 1
            erode = False
        else:
            main_func = cv2.erode
            scale = generator.uniform(*erode_scale_range)
            iterations = 1
            erode = True

        kernel_size = max(min(obj_bbox[1] - obj_bbox[0], obj_bbox[3] - obj_bbox[2]) / scale, 2) * 2 + 1
        kernel_size = int(kernel_size + 0.5)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        return main_func, iterations, kernel, erode

    max_attempts = 3
    success = False

    erode = None if erode_scale_range is not None else False
    result = work_area
    if orig_area > 0:
        for attempt in range(max_attempts):
            main_func, iterations, kernel, erode = _get_params(erode=erode)
            result = main_func(work_area, kernel, iterations=iterations)
            if erode:
                ratio = np.count_nonzero(result) / float(orig_area)
                if ratio > min_erode_ratio:
                    success = True
                    break
                erode_scale_range = (2 * erode_scale_range[0], 2 * erode_scale_range[1])
            else:
                success = True
                break

        if not success:
            _, iterations, kernel, erode = _get_params(erode=False)
            result = cv2.dilate(work_area, kernel, iterations=iterations)

    result = put_result_back(mask, result, metadata)

    return result


def leave_one_component(binary_mask):
    binary_mask = binary_mask.astype('uint8')

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    sizes = stats[:, -1]

    if len(sizes) < 2:
        return binary_mask, None

    max_label = 1
    max_size = sizes[1]
    max_centroid = centroids[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]
            max_centroid = centroids[i]

    largest_component = np.zeros(output.shape, dtype=np.uint8)
    largest_component[output == max_label] = 1
    return largest_component, max_centroid


def random_elastic_transform(mask, generator=np.random, alpha_range=(0.8, 1.2),
                             alpha_affine_range=(0.4, 0.6), sigma_range=(0.5, 0.75),
                             relative_to_object_size=True, shrink_to_object=True):
    if relative_to_object_size:
        xi, yi = np.where(mask)
        if len(xi) == 0:
            return mask
        object_min_size = min(xi.max() - xi.min(), yi.max() - yi.min())
        scale = 4 * object_min_size
    else:
        mask_min_size = min(mask.shape[:2])
        scale = mask_min_size
    alpha = scale * generator.uniform(*alpha_range) * 2
    alpha_affine = scale * generator.uniform(*alpha_affine_range) * 0.05
    sigma = scale * generator.uniform(*sigma_range) * 0.1

    working_area, metadata = get_working_area(mask, padding=4, shrink_to_object=shrink_to_object)

    transformed_mask = elastic_transform(working_area, alpha, sigma, alpha_affine, random_state=generator,
                                         border_mode=cv2.BORDER_CONSTANT, value=0)

    transformed_mask = put_result_back(mask, transformed_mask, metadata)

    return transformed_mask


def stretch_object_mask(mask, ratio):
    xi, yi = np.where(mask)
    if len(xi) == 0:
        return mask
    bbox = (xi.min(), yi.min(), xi.max(), yi.max())

    stretch_shift = (bbox[2] - bbox[0]) / 2 * (ratio - 1), (bbox[3] - bbox[1]) / 2 * (ratio - 1)
    stretched_bbox = (
        max(int(bbox[0] - stretch_shift[0]), 0),
        max(int(bbox[1] - stretch_shift[1]), 0),
        min(int(bbox[2] + stretch_shift[0]), mask.shape[0]),
        min(int(bbox[3] + stretch_shift[1]), mask.shape[1])
    )
    if 0 in (
        bbox[2] - bbox[0], bbox[3] - bbox[1],
        stretched_bbox[2] - stretched_bbox[0], stretched_bbox[3] - stretched_bbox[1]
    ):
        return mask

    resized_object = cv2.resize(
        mask[bbox[0]:bbox[2], bbox[1]:bbox[3]],
        (stretched_bbox[2] - stretched_bbox[0], stretched_bbox[3] - stretched_bbox[1])[::-1],
        interpolation=cv2.INTER_LINEAR
    )

    stretched_mask = np.zeros_like(mask)
    stretched_mask[stretched_bbox[0]:stretched_bbox[2], stretched_bbox[1]:stretched_bbox[3]] = resized_object

    return stretched_mask


def random_stretch(mask, generator=np.random, max_coeff=1.3, min_coeff=0.7):
    ratio = generator.rand() * (max_coeff - min_coeff) + min_coeff
    stretched_mask = stretch_object_mask(mask, ratio)
    return stretched_mask


def smooth(img, filter_size, filter_type, shrink_to_object=True):
    img = img.astype(np.uint8)

    if min(img.shape[:-1]) < filter_size:
        return img

    if filter_size < MIN_FILTER_SIZE:
        return img

    padding = int(filter_size / 2 + 0.5) + 1

    # h, w = img.shape[:2]
    # img_pad = np.zeros((h + 2 * padding, w + 2 * padding), np.uint8)
    # img_pad[padding:-padding, padding:-padding] = img

    work_area, metadata = get_working_area(img, padding, shrink_to_object)
    if filter_size > min(work_area.shape[:2]):
        filter_size = min(work_area.shape[:2])
        if filter_size % 2 == 0:
            filter_size = filter_size - 1

    if filter_type == 'median':
        # transformed_img = cv2.medianBlur(work_area, filter_size)
        transformed_img = cv2.GaussianBlur(work_area, (filter_size,) * 2, sigmaX=filter_size, sigmaY=filter_size)
    elif filter_type == 'gaussian':
        transformed_img = cv2.GaussianBlur(work_area, (filter_size, ) * 2, sigmaX=filter_size, sigmaY=filter_size)
    else:
        raise NotImplementedError(f"Filter is one of {FILTER_TYPES}")

    transformed_img = put_result_back(img, transformed_img, metadata)
    # transformed_img = transformed_img[padding:-padding, padding:-padding]

    return transformed_img


def random_smooth(mask, shrink_to_object=True, generator=np.random):
    xi, yi = np.where(mask)
    if len(xi) == 0:
        return mask
    object_min_size = min(xi.max() - xi.min(), yi.max() - yi.min())
    # max_ratio = MAX_SIZE_RATIO // 8
    min_ratio = 128
    max_ratio = 16

    filter_size = bounded_randint(
        object_min_size // min_ratio,
        object_min_size // max_ratio,
        min_value=1,
        generator=generator
    ) * 2 + 1
    filter_type = generator.choice(FILTER_TYPES)

    smooth_mask = smooth(mask, filter_size=filter_size, filter_type=filter_type, shrink_to_object=shrink_to_object)
    return smooth_mask


def random_shift(mask, max_shift=None, generator=np.random):
    xi, yi = np.where(mask)
    if len(xi) == 0:
        return mask
    bbox = (xi.min(), yi.min(), xi.max(), yi.max())
    object_size = np.array([bbox[2] - bbox[0], bbox[3] - bbox[1]])

    if max_shift is not None:
        max_shift = np.minimum(max_shift, object_size // MAX_SIZE_RATIO // 2)
    else:
        max_shift = object_size // MAX_SIZE_RATIO // 2

    pad_x = bounded_randint(
        0,
        max_shift[0],
        generator=generator
    )
    shift_x = pad_x * generator.choice([-1, 1])
    pad_y = bounded_randint(
        0,
        max_shift[1],
        generator=generator
    )
    shift_y = pad_y * generator.choice([-1, 1])

    shifted_mask = np.zeros((mask.shape[0] + 2 * pad_x, mask.shape[1] + 2 * pad_y))
    shifted_bbox = (
        bbox[0] + shift_x + pad_x,
        bbox[1] + shift_y + pad_y,
        bbox[2] + shift_x + pad_x,
        bbox[3] + shift_y + pad_y
    )
    shifted_mask[
        shifted_bbox[0]:shifted_bbox[2],
        shifted_bbox[1]:shifted_bbox[3]
    ] = mask[
        bbox[0]:bbox[2],
        bbox[1]:bbox[3]
    ]

    if pad_x != 0:
        shifted_mask = shifted_mask[pad_x:-pad_x]
    if pad_y != 0:
        shifted_mask = shifted_mask[:, pad_y:-pad_y]
    return shifted_mask


def random_convex_hull(mask, convex_prob, generator=np.random):
    if generator.rand() < convex_prob:
        convex_mask = convex_hull(mask)
        if convex_mask is not None:
            return convex_mask, True
    return mask, False


def convex_hull(mask):
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return None
    contours = np.vstack(contours)
    hull = cv2.convexHull(contours)
    convex_mask = np.zeros_like(mask, dtype=np.uint8)
    cv2.fillPoly(convex_mask, pts=[hull[:, 0]], color=int(mask.max()))
    return convex_mask


def is_nearly_convex(single_component_mask, hull_ratio=0.7):
    contours, hierarchy = cv2.findContours(
        single_component_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if len(contours) == 0:
        return False
    hull = cv2.convexHull(contours[0])
    hull_area = cv2.contourArea(hull)
    area = cv2.contourArea(contours[0])
    return area / (hull_area + 1e-6) > hull_ratio


def get_min_distance_between_masks(mask1, mask2):
    min_distance = 0
    indices1 = np.where(mask1)
    indices2 = np.where(mask2)
    if len(indices2[0]) and len(indices1[0]):
        indices1 = np.vstack(indices1)
        indices2 = np.vstack(indices2)
        min_distance = np.min(np.abs([
            indices1.min(1) - indices2.min(1),
            indices1.max(1) - indices2.max(1)
        ]), axis=0)
    return min_distance


def random_deformations(mask, shrink_to_object, decreased_shift, generator):
    transformed_mask = random_smooth(mask, shrink_to_object=shrink_to_object, generator=generator)

    convex_enough = is_nearly_convex(transformed_mask, 0.6)
    min_coeff = 0.75 if convex_enough else 0.9
    max_coeff = 1.2 if convex_enough else 1.1
    transformed_mask = random_stretch(transformed_mask, min_coeff=min_coeff, max_coeff=max_coeff, generator=generator)

    max_shift = get_min_distance_between_masks(transformed_mask, mask)
    max_shift = (max_shift + 1) * 2 if convex_enough else max_shift
    if decreased_shift:
        max_shift = max_shift / 4

    transformed_mask = random_shift(transformed_mask, max_shift=max_shift, generator=generator)
    # Boundary objects may have cornered edges.
    transformed_mask = random_smooth(transformed_mask, shrink_to_object=shrink_to_object, generator=generator)
    return transformed_mask


def remove_small_components(mask, small_thresh=0.05):
    mask = mask.astype(np.uint8)

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    sizes = stats[:, -1]
    if len(sizes) < 2:
        return mask

    sizes = sizes[1:]
    # total_area = (binary_mask > 0).sum()
    best_idx = np.argsort(sizes)[::-1]
    max_obj_area = sizes[best_idx[0]]

    final_mask = np.zeros_like(mask)
    for bidx in best_idx:
        ratio = float(sizes[bidx]) / max_obj_area
        if ratio > small_thresh:
            final_mask[output == bidx + 1] = 1

    return final_mask
