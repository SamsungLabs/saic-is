import cv2
import numpy as np
import torch

from isegm.utils.misc import find_connected_component


def get_next_points(pred, gt, points, generator, click_indx, pred_thresh=0.49):
    assert click_indx > 0
    pred = pred.cpu().numpy()[:, 0, :, :]
    gt = gt.cpu().numpy()[:, 0, :, :] > 0.5

    fn_mask = np.logical_and(gt, pred < pred_thresh)
    fp_mask = np.logical_and(np.logical_not(gt), pred > pred_thresh)

    fn_mask = np.pad(fn_mask, ((0, 0), (1, 1), (1, 1)), 'constant').astype(np.uint8)
    fp_mask = np.pad(fp_mask, ((0, 0), (1, 1), (1, 1)), 'constant').astype(np.uint8)
    num_points = points.size(1) // 2
    points = points.clone()

    for bindx in range(fn_mask.shape[0]):
        fn_mask_dt = cv2.distanceTransform(fn_mask[bindx], cv2.DIST_L2, 5)[1:-1, 1:-1]
        fp_mask_dt = cv2.distanceTransform(fp_mask[bindx], cv2.DIST_L2, 5)[1:-1, 1:-1]

        fn_max_dist = np.max(fn_mask_dt)
        fp_max_dist = np.max(fp_mask_dt)

        is_positive = fn_max_dist > fp_max_dist
        dt = fn_mask_dt if is_positive else fp_mask_dt
        inner_mask = dt > max(fn_max_dist, fp_max_dist) / 2.0
        coords = generator.generate_points(inner_mask, is_positive=is_positive, num_points=1)
        if len(coords) > 0:
            coords = coords[0]
            if is_positive:
                points[bindx, num_points - click_indx, 0] = float(coords[0])
                points[bindx, num_points - click_indx, 1] = float(coords[1])
                points[bindx, num_points - click_indx, 2] = float(click_indx)
            else:
                points[bindx, 2 * num_points - click_indx, 0] = float(coords[0])
                points[bindx, 2 * num_points - click_indx, 1] = float(coords[1])
                points[bindx, 2 * num_points - click_indx, 2] = float(click_indx)
    return points


def get_next_contour_mask(pred, gt, generator, pred_thresh=0.49, dt_size=5, inner_mask_coeff=0.75):
    device = gt.device
    pred = pred.cpu().numpy()[:, 0, :, :]
    gt = gt.cpu().numpy()[:, 0, :, :] > 0.5

    fn_mask = np.logical_and(gt, pred < pred_thresh).astype(np.uint8)
    fp_mask = np.logical_and(np.logical_not(gt), pred > pred_thresh).astype(np.uint8)
    contours = np.zeros((gt.shape[0], 2, gt.shape[1], gt.shape[2]), dtype=pred.dtype)
    mask_for_interactive_loss = np.zeros_like(contours)
    for bindx in range(fn_mask.shape[0]):
        mask = gt[bindx]
        if fn_mask[bindx].sum() == 0 or fp_mask[bindx].sum() == 0:
            fn_mask[bindx] = np.logical_and(mask, pred[bindx] < pred_thresh * 0.6).astype(np.uint8)
            fp_mask[bindx] = np.logical_and(np.logical_not(mask), pred[bindx] > pred_thresh * 0.6).astype(np.uint8)

        diff_map, is_positive = get_contour_generation_params(
            mask, fn_mask[bindx], fp_mask[bindx],
            dt_size, inner_mask_coeff
        )
        if diff_map is None:
            diff_map = np.zeros(mask.shape[:2], dtype=np.float32)
            contour_mask = np.zeros((2, mask.shape[0], mask.shape[1]), dtype=np.float32)
        else:
            contour_mask = generator.generate_contour_mask(diff_map.copy(), is_positive)

        contours[bindx] = contour_mask
        if is_positive:
            mask_for_interactive_loss[bindx][0] = diff_map
        else:
            mask_for_interactive_loss[bindx][1] = diff_map
    return torch.tensor(contours, device=device), torch.tensor(mask_for_interactive_loss, device=device)


def get_contour_generation_params(mask, fn_mask, fp_mask, dt_size, inner_mask_coeff):
    fn_mask_dt = cv2.distanceTransform(fn_mask, cv2.DIST_L2, dt_size)
    fp_mask_dt = cv2.distanceTransform(fp_mask, cv2.DIST_L2, dt_size)
    fn_max_dist = np.max(fn_mask_dt)
    fp_max_dist = np.max(fp_mask_dt)
    is_positive = fn_max_dist > fp_max_dist

    if is_positive:
        inner_mask = fn_mask_dt > fn_max_dist * inner_mask_coeff
        dt = fn_mask_dt
    else:
        inner_mask = fp_mask_dt > fp_max_dist * inner_mask_coeff
        dt = fp_mask_dt
    indices = np.argwhere(inner_mask)
    if len(indices) > 0:
        dt = find_connected_component(indices, dt)
        dt = dt > np.max(dt) * 0.2
    else:
        dt = None

    return dt, is_positive


def get_next_strokes(pred, gt, generator, pred_thresh=0.49):
    device = gt.device
    pred = pred.cpu().numpy()[:, 0, :, :]
    gt = gt.cpu().numpy()[:, 0, :, :] > 0.5

    strokes = np.zeros((gt.shape[0], 2, gt.shape[1], gt.shape[2]), dtype=pred.dtype)
    stroke_ends = np.zeros((gt.shape[0], 4, 3), dtype=np.int) - 1
    for bindx in range(gt.shape[0]):
        fn_mask = np.logical_and(gt[bindx], pred[bindx] < pred_thresh).astype(np.uint8)
        fp_mask = np.logical_and(np.logical_not(gt[bindx]), pred[bindx] > pred_thresh).astype(np.uint8)
        if fn_mask.sum() == 0 or fp_mask.sum() == 0:
            fn_mask = np.logical_and(gt[bindx], pred[bindx] < pred_thresh * 0.6).astype(np.uint8)
            fp_mask = np.logical_and(np.logical_not(gt[bindx]), pred[bindx] > pred_thresh * 0.6).astype(np.uint8)

        fn_mask_dt = cv2.distanceTransform(fn_mask, cv2.DIST_L2, 5)
        fp_mask_dt = cv2.distanceTransform(fp_mask, cv2.DIST_L2, 5)
        fn_max_dist = np.max(fn_mask_dt)
        fp_max_dist = np.max(fp_mask_dt)
        is_positive = fn_max_dist > fp_max_dist
        if is_positive:
            mask = fn_mask
        else:
            mask = fp_mask
        stroke_mask, metadata = generator.generate_stroke_mask(mask, is_positive=is_positive, with_meta=True)
        strokes[bindx] = stroke_mask
        x0, y0, x1, y1 = metadata['ends']
        class_i = 0 if is_positive else 2
        stroke_ends[bindx][class_i] = x0, y0, int(x0 >= 0 and y0 >= 0) - 1
        stroke_ends[bindx][class_i + 1] = x1, y1, int(x1 >= 0 and y1 >= 0) - 1
    return torch.tensor(strokes, device=device), torch.tensor(stroke_ends, device=device)
