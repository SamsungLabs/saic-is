from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np
import torch

from isegm.data.datasets import (
    BerkeleyDataset,
    DavisDataset,
    GrabCutDataset,
    PascalVocDataset,
    SBDEvaluationDataset,
    SAICDataset
)
from isegm.data.interaction_type import IType
from isegm.utils.misc import mask_to_boundary, get_boundary_size
from isegm.utils.serialization import load_model


def get_time_metrics(all_ious, elapsed_time):
    n_images = len(all_ious)
    n_clicks = sum(map(len, all_ious))

    mean_spc = elapsed_time / n_clicks
    mean_spi = elapsed_time / n_images

    return mean_spc, mean_spi


def load_is_model(checkpoint, device, **kwargs):
    if isinstance(checkpoint, (str, Path)):
        state_dict = torch.load(checkpoint, map_location='cpu')
    else:
        state_dict = checkpoint

    if isinstance(state_dict, list):
        model = load_single_is_model(state_dict[0], device, **kwargs)
        models = [load_single_is_model(x, device, **kwargs) for x in state_dict]

        return model, models
    else:
        return load_single_is_model(state_dict, device, **kwargs)


def load_single_is_model(state_dict, device, **kwargs):
    model = load_model(state_dict['config'], **kwargs)
    model.load_state_dict(state_dict['state_dict'], strict=False)

    for param in model.parameters():
        param.requires_grad = False
    model.to(device)
    model.eval()

    return model


def get_dataset(dataset_name, cfg, input_type=IType.point):
    if dataset_name == 'GrabCut':
        dataset = GrabCutDataset(cfg.GRABCUT_PATH, input_type=input_type)
    elif dataset_name == 'Berkeley':
        dataset = BerkeleyDataset(cfg.BERKELEY_PATH, input_type=input_type)
    elif dataset_name == 'DAVIS':
        dataset = DavisDataset(cfg.DAVIS_PATH, input_type=input_type)
    elif dataset_name == 'SBD':
        dataset = SBDEvaluationDataset(cfg.SBD_PATH, input_type=input_type)
    elif dataset_name == 'SBD_Train':
        dataset = SBDEvaluationDataset(cfg.SBD_PATH, split='train', input_type=input_type)
    elif dataset_name == 'PascalVOC':
        dataset = PascalVocDataset(cfg.PASCALVOC_PATH, split='test', input_type=input_type)
    elif dataset_name == 'COCO_MVal':
        dataset = DavisDataset(cfg.COCO_MVAL_PATH, input_type=input_type)
    elif dataset_name == 'SAIC_IS':
        dataset = SAICDataset(cfg.SAIC_IS_PATH, input_type=input_type, max_side_size=-1)
    else:
        dataset = None

    return dataset


def get_iou(gt_mask, pred_mask, ignore_label=-1):
    ignore_gt_mask_inv = gt_mask != ignore_label
    obj_gt_mask = gt_mask == 1

    intersection = np.logical_and(np.logical_and(pred_mask, obj_gt_mask), ignore_gt_mask_inv).sum()
    union = np.logical_and(np.logical_or(pred_mask, obj_gt_mask), ignore_gt_mask_inv).sum()

    return intersection / union


def compute_noi_metric(all_ious, iou_thrs, max_interactions=20):
    max_len = max(map(len, all_ious))
    all_ious = [np.concatenate([x, [x[-1]] * (max_len - len(x))]) for x in all_ious]
    assert np.all([len(x) == len(all_ious[0]) for x in all_ious])
    all_ious = np.array(all_ious)
    iou_thrs = np.array(iou_thrs)
    iou_over_thr = all_ious[:, None, :] >= iou_thrs[None, :, None]  # N x n_thrs x max_interactions
    scores_arr = np.ones((len(iou_over_thr), len(iou_thrs)), dtype=np.int) * max_interactions  # N x n_thrs
    any_over_thr = np.any(iou_over_thr, axis=2)  # N x n_thrs
    scores_arr[any_over_thr] = np.argmax(iou_over_thr, axis=2)[any_over_thr] + 1
    noi_list = scores_arr.mean(axis=0)  # n_thrs
    over_max_list = (scores_arr == max_interactions).sum(axis=0)  # n_thrs
    return noi_list, over_max_list


def compute_mean_spo_dict(all_times):
    mean_spo = {}
    time_per_operation = defaultdict(list)
    for time_dict in all_times:
        for k, v in time_dict.items():
            time_per_operation[k].append(v)
    for k, v in time_per_operation.items():
        mean_spo[k] = sum(v) / len(v)
    return mean_spo


def find_checkpoint(weights_folder, checkpoint_name):
    weights_folder = Path(weights_folder)
    if ':' in checkpoint_name:
        model_name, checkpoint_name = checkpoint_name.split(':')
        models_candidates = [x for x in weights_folder.glob(f'{model_name}*') if x.is_dir()]
        assert len(models_candidates) == 1
        model_folder = models_candidates[0]
    else:
        model_folder = weights_folder

    if checkpoint_name.endswith('.pth'):
        if Path(checkpoint_name).exists():
            checkpoint_path = checkpoint_name
        else:
            checkpoint_path = weights_folder / checkpoint_name
    else:
        model_checkpoints = list(model_folder.rglob(f'{checkpoint_name}*.pth'))
        assert len(model_checkpoints) == 1
        checkpoint_path = model_checkpoints[0]

    return str(checkpoint_path)


def get_noi_table(noi_list, over_max_list, dataset_name, mean_spc, elapsed_time,
                  n_clicks=20, model_name=None, mean_spo=None):
    table_header = (
        f'|{"Dataset":^15}|'
        f'{"NoI@80%":^9}|{"NoI@85%":^9}|{"NoI@90%":^9}|'
        f'{">=" + str(n_clicks) + "@85%":^9}|{">=" + str(n_clicks) + "@90%":^9}|'
        f'{"SPC,s":^7}|{"Time":^9}|'
    )
    if mean_spo:
        table_header = f'{table_header}{"mSP_gen":^9}|{"mSP_pred":^9}|{"mSP_iou":^9}|'
    row_width = len(table_header)

    header = f'Eval results for model: {model_name}\n' if model_name is not None else ''
    header += '-' * row_width + '\n'
    header += table_header + '\n' + '-' * row_width

    eval_time = str(timedelta(seconds=int(elapsed_time)))
    table_row = f'|{dataset_name:^15}|'
    table_row += f'{noi_list[0]:^9.2f}|'
    table_row += f'{noi_list[1]:^9.2f}|' if len(noi_list) > 1 else f'{"?":^9}|'
    table_row += f'{noi_list[2]:^9.2f}|' if len(noi_list) > 2 else f'{"?":^9}|'
    table_row += f'{over_max_list[1]:^9}|' if len(noi_list) > 1 else f'{"?":^9}|'
    table_row += f'{over_max_list[2]:^9}|' if len(noi_list) > 2 else f'{"?":^9}|'
    table_row += f'{mean_spc:^7.3f}|{eval_time:^9}|'
    if mean_spo:
        table_row += f'{mean_spo["interaction"] * 1000:^9.3f}|'
        table_row += f'{mean_spo["prediction"] * 1000:^9.3f}|'
        table_row += f'{mean_spo["iou"] * 1000:^9.3f}|'

    return header, table_row


def get_avg_table(avg_iou, std_iou, dataset_name, mean_spc, elapsed_time,
                  model_name=None, mean_spo=None):
    table_header = (
        f'|{"Dataset":^15}|'
        f'{"AvgIoU":^9}|{"StdIoU":^9}|'
        f'{"SPC,s":^7}|{"Time":^9}|'
    )
    if mean_spo:
        table_header = f'{table_header}{"mSP_gen":^9}|{"mSP_pred":^9}|{"mSP_iou":^9}|'
    row_width = len(table_header)

    header = f'Eval results for model: {model_name}\n' if model_name is not None else ''
    header += '-' * row_width + '\n'
    header += table_header + '\n' + '-' * row_width

    eval_time = str(timedelta(seconds=int(elapsed_time)))
    table_row = f'|{dataset_name:^15}|'
    table_row += f'{avg_iou:^9.2f}|'
    table_row += f'{std_iou:^9.2f}|'
    table_row += f'{mean_spc:^7.3f}|{eval_time:^9}|'
    if mean_spo:
        table_row += f'{mean_spo["interaction"] * 1000:^9.3f}|'
        table_row += f'{mean_spo["prediction"] * 1000:^9.3f}|'
        table_row += f'{mean_spo["iou"] * 1000:^9.3f}|'
    return header, table_row


def get_fn_fp_distance_transform(pred_mask, gt_mask, not_clicked_map, not_ignore_map, padding):
    fn_mask = np.logical_and(np.logical_and(gt_mask, np.logical_not(pred_mask)), not_ignore_map)
    fp_mask = np.logical_and(np.logical_and(np.logical_not(gt_mask), pred_mask), not_ignore_map)
    if padding:
        fn_mask = np.pad(fn_mask, ((1, 1), (1, 1)), 'constant')
        fp_mask = np.pad(fp_mask, ((1, 1), (1, 1)), 'constant')
    fn_mask_dt = cv2.distanceTransform(fn_mask.astype(np.uint8), cv2.DIST_L2, 0)
    fp_mask_dt = cv2.distanceTransform(fp_mask.astype(np.uint8), cv2.DIST_L2, 0)
    if padding:
        fn_mask_dt = fn_mask_dt[1:-1, 1:-1]
        fp_mask_dt = fp_mask_dt[1:-1, 1:-1]
        fn_mask = fn_mask[1:-1, 1:-1]
        fp_mask = fp_mask[1:-1, 1:-1]
    fn_mask_dt = fn_mask_dt * not_clicked_map
    fp_mask_dt = fp_mask_dt * not_clicked_map
    return fn_mask, fn_mask_dt, fp_mask, fp_mask_dt


def get_boundary_iou(gt_mask, pred_mask, ignore_label=-1, dilation_ratio=0.005, use_mask_diag=False):
    ignore_gt_mask_inv = gt_mask != ignore_label
    obj_gt_mask = gt_mask == 1

    boundary_size = get_boundary_size(obj_gt_mask, dilation_ratio=dilation_ratio, use_mask_diag=use_mask_diag)
    obj_gt_mask = mask_to_boundary(obj_gt_mask.astype(np.uint8), boundary_size=boundary_size)
    pred_mask = mask_to_boundary(pred_mask.astype(np.uint8), boundary_size=boundary_size)

    intersection = np.logical_and(np.logical_and(pred_mask, obj_gt_mask), ignore_gt_mask_inv).sum()
    union = np.logical_and(np.logical_or(pred_mask, obj_gt_mask), ignore_gt_mask_inv).sum()
    return intersection / union
