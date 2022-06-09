import torch
import numpy as np
import cv2

from isegm.utils.log import logger


def get_dims_with_exclusion(dim, exclude=None):
    dims = list(range(dim))
    if exclude is not None:
        dims.remove(exclude)

    return dims


def save_checkpoint(net, checkpoints_path, epoch=None, prefix='', verbose=True, multi_gpu=False):
    if epoch is None:
        checkpoint_name = 'last_checkpoint.pth'
    else:
        checkpoint_name = f'{epoch:03d}.pth'

    if prefix:
        checkpoint_name = f'{prefix}_{checkpoint_name}'

    if not checkpoints_path.exists():
        checkpoints_path.mkdir(parents=True)

    checkpoint_path = checkpoints_path / checkpoint_name
    if verbose:
        logger.info(f'Save checkpoint to {str(checkpoint_path)}')

    net = net.module if multi_gpu else net
    torch.save({'state_dict': net.state_dict(),
                'config': net._config}, str(checkpoint_path))


def get_bbox_from_mask(mask):
    if not np.any(mask):
        mask[0, 0] = 1
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


def expand_bbox(bbox, expand_ratio, min_crop_size=None):
    rmin, rmax, cmin, cmax = bbox
    rcenter = 0.5 * (rmin + rmax)
    ccenter = 0.5 * (cmin + cmax)
    height = expand_ratio * (rmax - rmin + 1)
    width = expand_ratio * (cmax - cmin + 1)
    if min_crop_size is not None:
        height = max(height, min_crop_size)
        width = max(width, min_crop_size)

    rmin = int(round(rcenter - 0.5 * height))
    rmax = int(round(rcenter + 0.5 * height))
    cmin = int(round(ccenter - 0.5 * width))
    cmax = int(round(ccenter + 0.5 * width))

    return rmin, rmax, cmin, cmax


def clamp_bbox(bbox, rmin, rmax, cmin, cmax):
    return (max(rmin, bbox[0]), min(rmax, bbox[1]),
            max(cmin, bbox[2]), min(cmax, bbox[3]))


def get_bbox_iou(b1, b2):
    h_iou = get_segments_iou(b1[:2], b2[:2])
    w_iou = get_segments_iou(b1[2:4], b2[2:4])
    return h_iou * w_iou


def get_segments_iou(s1, s2):
    a, b = s1
    c, d = s2
    intersection = max(0, min(b, d) - max(a, c) + 1)
    union = max(1e-6, max(b, d) - min(a, c) + 1)
    return intersection / union


def get_labels_with_sizes(x):
    obj_sizes = np.bincount(x.flatten())
    labels = np.nonzero(obj_sizes)[0].tolist()
    labels = [x for x in labels if x != 0]
    return labels, obj_sizes[labels].tolist()


def get_labels_with_rois(x):
    values = np.unique(x.flatten())
    values = [v for v in values if v > 0]
    bboxes = [get_bbox_from_mask(x == v) for v in values]

    return values, bboxes


def bounded_randint(left, right, min_value=None, max_value=None, generator=np.random):
    random_value = generator.randint(left, max(left + 1, right))
    if max_value is not None:
        random_value = min(random_value, max_value)
    if min_value is not None:
        random_value = max(random_value, min_value)
    return random_value


def get_mask_rectangle(mask):
    col = np.argwhere(mask.sum(axis=1))
    if len(col) == 0:
        col = [0]
    min_col, max_col = int(col[0]), int(col[-1])
    row = np.argwhere(mask.sum(axis=0))
    if len(row) == 0:
        row = [0]
    min_row, max_row = int(row[0]), int(row[-1])
    h = max_col - min_col + 1
    w = max_row - min_row + 1
    return (min_col + max_col) // 2, (min_row + max_row) // 2, h, w


def get_gaussian_1d_numpy(coord, shape, sigma):
    x = np.arange(-coord, shape - coord)
    denom = np.sqrt(2 * np.pi * sigma)
    return np.exp(-0.5 * ((x / sigma) ** 2)) / denom


def get_gaussian_numpy(coords, gt_mask):
    _, _, h, w = get_mask_rectangle(gt_mask)
    x = get_gaussian_1d_numpy(coords[0], gt_mask.shape[0], h / 2)
    y = get_gaussian_1d_numpy(coords[1], gt_mask.shape[1], w / 2)
    return x[:, None] * y[None, :]


def get_gaussian_1d_torch(coords, shape, sigmas, device):
    x = torch.arange(shape, device=device)  # coords: B, Cnt, 5
    x = x.view(1, 1, x.size(0))
    sigma_calc = sigmas / 8
    ans = torch.exp(-0.5 * (((x - coords[:, :, None]) / sigma_calc[:, :, None]) ** 2))
    ans[(x < (coords - sigmas)[:, :, None]) | (x > (coords + sigmas)[:, :, None])] = 0
    return ans


def get_gaussian_torch(coords, pred):
    x = get_gaussian_1d_torch(coords[:, :, 0], pred.shape[2], coords[:, :, 2] / 2, pred.device)
    y = get_gaussian_1d_torch(coords[:, :, 1], pred.shape[3], coords[:, :, 3] / 2, pred.device)
    return x[:, :, :, None] * y[:, :, None, :]


def find_connected_component(nonzero_indices, connectivity_map):
    row, col = nonzero_indices[np.random.randint(0, len(nonzero_indices))]
    connectivity_map[row][col:] = np.minimum.accumulate(connectivity_map[row][col:])
    connectivity_map[row][:col][::-1] = np.minimum.accumulate(connectivity_map[row][:col][::-1])
    connectivity_map[row:] = np.minimum.accumulate(connectivity_map[row:], axis=0)
    connectivity_map[:row][::-1] = np.minimum.accumulate(connectivity_map[:row][::-1], axis=0)
    connectivity_map[:, col:] = np.minimum.accumulate(connectivity_map[:, col:], axis=1)
    connectivity_map[:, :col][:, ::-1] = np.minimum.accumulate(connectivity_map[:, :col][:, ::-1], axis=1)
    gauss = get_gaussian_numpy([row, col], connectivity_map > 0)
    connectivity_map = connectivity_map * gauss
    return connectivity_map


def mask_to_boundary(mask, boundary_size, with_erode=False):
    """
    Convert binary mask to boundary mask.
        mask: numpy.array(uint8), binary mask
        boundary_size: float, dilation size
        with_erode: boolean
            if the mask without the boundary should be returned along with the boundary mask

    return:
        boundary mask: numpy.array
    """
    h, w = mask.shape
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=boundary_size)
    mask_erode = new_mask_erode[1:h + 1, 1:w + 1]
    # G_d intersects G in the paper.
    if with_erode:
        return mask - mask_erode, mask_erode
    return mask - mask_erode


def get_boundary_size(mask, dilation_ratio=0.02, use_mask_diag=False):
    if dilation_ratio < 1:
        bbox = get_bbox_from_mask(mask) if use_mask_diag else (1, mask.shape[0], 1, mask.shape[1])
        bbox_h = bbox[1] - bbox[0] + 1
        bbox_w = bbox[3] - bbox[2] + 1
        diag_size = np.sqrt(bbox_h ** 2 + bbox_w ** 2)
        dilation = int(round(dilation_ratio * diag_size))
    else:
        dilation = int(round(dilation_ratio))
    if dilation < 1:
        dilation = 1
    return dilation


def get_iou(gt_mask, pred_mask, ignore_label=-1, gt_relative=False):
    ignore_gt_mask_inv = gt_mask != ignore_label
    obj_gt_mask = gt_mask == 1

    intersection = np.logical_and(np.logical_and(pred_mask, obj_gt_mask), ignore_gt_mask_inv).sum()
    if gt_relative:
        union = gt_mask.sum()
    else:
        union = np.logical_and(np.logical_or(pred_mask, obj_gt_mask), ignore_gt_mask_inv).sum()

    return intersection / (1e-6 + union)


def limit_longest_size(image, max_side_size, target_size=None, interpolation=cv2.INTER_LINEAR):
    if target_size is None:
        if image.shape[0] > image.shape[1]:
            target_h = min(image.shape[0], max_side_size)
            target_w = int(image.shape[1] / image.shape[0] * target_h + 0.5)
        else:
            target_w = min(image.shape[1], max_side_size)
            target_h = int(image.shape[0] / image.shape[1] * target_w + 0.5)
    else:
        target_h, target_w = target_size
    dtype = image.dtype
    image = cv2.resize(image.astype(np.uint8), (target_w, target_h), interpolation=interpolation)
    image = image.astype(dtype)
    return image, (target_h, target_w)
