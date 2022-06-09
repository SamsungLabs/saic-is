import argparse
from collections import defaultdict
from pathlib import Path
import pickle
import sys

import cv2
import torch
import numpy as np

sys.path.insert(0, '.')
from isegm.data.interaction_type import (
    IType,
    InteractionSelector, ProductInteractionSelector, SingleInteractionSelector
)
from isegm.inference import utils
from isegm.inference.evaluation import evaluate_dataset
from isegm.inference.predictors import get_predictor
from isegm.utils.exp import load_config_file
from isegm.utils.log import logger, add_logging
from isegm.utils.vis import draw_probmap, draw_with_blend_and_clicks, draw_with_blend_and_lines


def parse_args():
    parser = argparse.ArgumentParser()

    group_checkpoints = parser.add_mutually_exclusive_group(required=True)
    group_checkpoints.add_argument('--exp-path', type=str, default='',
                                   help='Path(s) to experiment(s) (relative to cfg.EXPS_PATH)'
                                        ' in format relative_path:checkpoint_prefix.')
    parser.add_argument('--exp-checkpoint-id', type=str, default=None)

    parser.add_argument('--datasets', type=str, default='GrabCut,Berkeley,DAVIS,SBD,PascalVOC,SAIC_IS',
                        help='List of datasets on which the model should be tested. '
                             'Datasets are separated by a comma. Possible choices: '
                             'GrabCut, Berkeley, DAVIS, SBD, PascalVOC, SAIC_IS')

    group_device = parser.add_mutually_exclusive_group()
    group_device.add_argument('--gpus', type=str, default='0',
                              help='ID of used GPU.')
    group_device.add_argument('--cpu', action='store_true', default=False,
                              help='Use only CPU for inference.')

    group_iou_thresh = parser.add_mutually_exclusive_group()
    group_iou_thresh.add_argument('--target-iou', type=float, default=0.90,
                                  help='Target IoU threshold for the NoI metric. (min possible value = 0.8)')
    group_iou_thresh.add_argument('--iou-analysis', action='store_true', default=False,
                                  help='Plot mIoU(number of interactions) with target_iou=1.0.')

    parser.add_argument('--n-interactions', type=int, default=10,
                        help='Maximum number of interactions for the NoI metric.')
    parser.add_argument('--min-n-interactions', type=int, default=1,
                        help='Minimum number of interactions for the evaluation.')
    parser.add_argument('--thresh', type=float, required=False, default=0.49,
                        help='The segmentation mask is obtained from the probability outputs using this threshold.')
    parser.add_argument('--interactions-limit', type=int, default=None)

    parser.add_argument('--print', action='store_true', default=False)
    parser.add_argument('--save-ious', action='store_true', default=False)
    parser.add_argument('--log-model', action='store_true', default=False)
    parser.add_argument('--vis-preds', action='store_true', default=False)
    parser.add_argument('--separate-dirs', action='store_true', default=False,
                        help='Save samples for each image to separate directory')
    parser.add_argument('--config-path', type=str, default='./config.yml',
                        help='The path to the config file.')
    parser.add_argument('--logs-path', type=str, default='',
                        help='The path to the evaluation logs. Default path: cfg.EXPS_PATH/evaluation_logs.')
    parser.add_argument('--fixed-h', type=int, default=None, help='Fixed height.')
    parser.add_argument('--fixed-w', type=int, default=None, help='Fixed width.')
    parser.add_argument('--max-size', type=int, default=None, help='Maximum size of the longest side of an image')
    parser.add_argument('--no-zoomin', action='store_true', default=False)
    parser.add_argument('--boundary-iou', action='store_true', default=False)
    parser.add_argument('--boundary-size', type=float, default=0.005,
                        help='Size of the boundary when computing boundary-iou, if boundary-iou < 1 then '
                             'boundary_size_px=max_diagonal_px*boundary-size, if boundary-iou >=1 then'
                             'boundary_size_px=int(boundary-size).')
    parser.add_argument('--with-spo', action='store_true', default=False,
                        help='Print seconds per operation: interaction generation, prediction, metric computation')
    parser.add_argument('--noi', action='store_true', default=False,
                        help='Compute NoI@[85, 90, 95] instead of IoU for all possible interaction combinations at 3 first steps.')

    args = parser.parse_args()
    args.ui_product = ~args.noi
    if args.cpu:
        args.device = torch.device('cpu')
    else:
        args.device = torch.device(f"cuda:{args.gpus.split(',')[0]}")

    if (args.iou_analysis or args.ui_product) and args.min_n_interactions <= 1:
        args.target_iou = 1.01
    else:
        args.target_iou = max(0.8, args.target_iou)

    cfg = load_config_file(args.config_path, return_edict=True)
    cfg.EXPS_PATH = Path(cfg.EXPS_PATH)

    if args.logs_path == '':
        args.logs_path = cfg.EXPS_PATH / 'evaluation_logs'
    else:
        args.logs_path = Path(args.logs_path)
    return args, cfg


def main():
    args, cfg = parse_args()

    checkpoints_list, logs_path, logs_prefix = get_checkpoints_list_and_logs_path(args, cfg)
    logs_path.mkdir(parents=True, exist_ok=True)
    if not args.print:
        add_logging(logs_path, prefix='eval_')

    models_by_ui = {}
    for checkpoint_path in checkpoints_list:
        model = utils.load_is_model(checkpoint_path, args.device)
        if args.log_model:
            log_model_type(model)
        input_type = [IType.point]
        if hasattr(model, 'input_type') and model.input_type:
            if isinstance(model.input_type, str):
                model.input_type = IType[model.input_type]
            if isinstance(model.input_type, list):
                input_type = [IType[itype.name] for itype in model.input_type]
            else:
                input_type = [IType[model.input_type.name]]
        for itype in input_type:
            assert itype not in models_by_ui
            models_by_ui[itype] = model

    print_header = True
    print_function = print if args.print else logger.info
    printed_table = '\n'
    for dataset_name in args.datasets.split(','):
        dataset = utils.get_dataset(dataset_name, cfg)
        predictor_params, zoomin_params = get_predictor_and_zoomin_params(args, dataset_name)
        fixed_size = None
        if args.fixed_h is not None:
            fixed_size = (args.fixed_h, args.fixed_w)
        if args.no_zoomin:
            zoomin_params = None
        predictor = get_predictor(models_by_ui, args.device,
                                  predictor_params=predictor_params,
                                  zoom_in_params=zoomin_params,
                                  fixed_size=fixed_size)
        if len(models_by_ui) == 1:
            interaction_selector = SingleInteractionSelector(list(models_by_ui.keys())[0])
        elif args.ui_product:
            interaction_selector = ProductInteractionSelector()
        else:
            interaction_selector = InteractionSelector((
                (IType.contour, (0, 1)),
                (IType.stroke, (1, 3)),
                (IType.point, (3, 1000))
            ))
        vis_callback = (
            get_prediction_vis_callback(logs_path, dataset_name, args.thresh, args.separate_dirs)
            if args.vis_preds else None
        )
        dataset_results = evaluate_dataset(
            dataset, predictor, pred_thr=args.thresh, max_iou_thr=args.target_iou,
            min_interactions=args.min_n_interactions, max_interactions=args.n_interactions,
            callback=vis_callback, interaction_selector=interaction_selector,
            boundary_iou=args.boundary_iou, boundary_size=args.boundary_size,
            contour_filled=cfg.MODEL.CONTOUR_FILLED, max_size=args.max_size,
            logger=None if args.print else logger
        )

        if args.iou_analysis:
            save_iou_analysis_data(args, dataset_name, logs_path,
                                   logs_prefix, dataset_results,
                                   interaction_selector=interaction_selector)

        if args.ui_product:
            printed = save_average_results(
                args, dataset_name, logs_path, logs_prefix, dataset_results,
                print_header=print_header, with_times=args.with_spo,
                print_function=print_function
            )
        else:
            printed = save_noi_results(
                args, dataset_name, logs_path, logs_prefix, dataset_results,
                save_ious=args.save_ious, single_model_eval=True,
                print_header=print_header, with_times=args.with_spo,
                print_function=print_function
            )
        print_header = False
        printed_table = printed_table + printed + '\n'
    if not args.print:
        print_function(printed_table)


def log_model_type(model):
    params_str = ''
    for attr in ['with_prev_mask', 'input_type']:
        attr_value = getattr(model, attr, None)
        if attr_value is not None:
            params_str += f'\n{attr}: {attr_value}'
    if params_str:
        logger.info(f'Model params:{params_str}')


def get_predictor_and_zoomin_params(args, dataset_name):
    predictor_params = {}

    if args.interactions_limit is not None:
        if args.interactions_limit == -1:
            args.interactions_limit = args.n_interactions
        predictor_params['net_interactions_limit'] = args.interactions_limit

    zoom_in_params = {
        'target_size': 600 if dataset_name == 'DAVIS' else 400,
        'skip_interactions': {
            IType.point: 1,
            IType.contour: -1,
            IType.stroke: -1,
        }
    }

    return predictor_params, zoom_in_params


def get_checkpoints_list_and_logs_path(args, cfg):
    logs_prefix = ''
    exp_paths = args.exp_path.split(',')
    ch_paths = []
    for rel_exp_path in exp_paths:
        checkpoint_prefix = ''
        if ':' in rel_exp_path:
            rel_exp_path, checkpoint_prefix = rel_exp_path.split(':')

        exp_path_prefix = cfg.EXPS_PATH / rel_exp_path
        candidates = list(exp_path_prefix.parent.glob(exp_path_prefix.stem + '*'))
        exp_path = candidates[0]
        checkpoints_list = sorted((exp_path / 'checkpoints').glob(checkpoint_prefix + '*.pth'), reverse=True)
        assert len(checkpoints_list) > 0, "Couldn't find any checkpoints."

        if args.exp_checkpoint_id is not None:
            checkpoints_list = [exp_path / 'checkpoints' / f'{args.exp_checkpoint_id}.pth']
            logs_prefix = f'checkp_{args.exp_checkpoint_id}'
        elif checkpoint_prefix:
            if len(checkpoints_list) == 1:
                logs_prefix = checkpoints_list[0].stem
            else:
                logs_prefix = f'all_{checkpoint_prefix}'
        else:
            logs_prefix = 'all_checkpoints'

        ch_paths += checkpoints_list
    if len(ch_paths) > 1:
        logs_path = args.logs_path / 'merged_models' / '-'.join([str(path).split('/')[-3] for path in ch_paths])
    else:
        logs_path = args.logs_path / exp_path.relative_to(cfg.EXPS_PATH)
    return ch_paths, logs_path, logs_prefix


def save_average_results(
    args, dataset_name, logs_path, logs_prefix, dataset_results,
    print_header=True, with_times=False, print_function=print
):
    all_ious, elapsed_time, all_times = dataset_results
    mean_spc, mean_spi = utils.get_time_metrics(all_ious[0], elapsed_time)
    mean_spo = utils.compute_mean_spo_dict(all_times) if with_times else {}
    assert np.all([len(x) == len(all_ious[0]) for x in all_ious])
    all_ious = np.array(all_ious)  # N x 27 x n_interactions
    product_curves = all_ious.mean(axis=0)  # 27 x n_interactions
    product_averages = product_curves.mean(axis=1) * 100  # 27
    logger.info(
        f'Ious array shape: {all_ious.shape}.'
        f' Curves shape: {product_curves.shape}.'
        f' Curve averages shape: {product_averages.shape}'
    )
    metric_average = product_averages.mean()
    metric_std = product_averages.std()

    model_name = str(logs_path.relative_to(args.logs_path)) + ':' + logs_prefix if logs_prefix else logs_path.stem
    header, table_row = utils.get_avg_table(
        metric_average, metric_std, dataset_name, mean_spc, elapsed_time,
        model_name=model_name, mean_spo=mean_spo
    )
    printed = table_row
    if print_header:
        printed = header + '\n' + printed
        print_function(header)
    print_function(table_row)

    name_prefix = ''
    if logs_prefix:
        name_prefix = logs_prefix + '_'

    log_path = logs_path / f'{name_prefix}{args.n_interactions}_avg.txt'
    if log_path.exists():
        with open(log_path, 'a') as f:
            f.write(table_row + '\n')
    else:
        with open(log_path, 'w') as f:
            if print_header:
                f.write(header + '\n')
            f.write(table_row + '\n')
    return printed


def save_noi_results(
    args, dataset_name, logs_path, logs_prefix, dataset_results,
    save_ious=False, print_header=True, single_model_eval=False, with_times=False,
    print_function=print
):
    all_ious, elapsed_time, all_times = dataset_results
    mean_spc, mean_spi = utils.get_time_metrics(all_ious, elapsed_time)
    mean_spo = utils.compute_mean_spo_dict(all_times) if with_times else {}

    iou_thrs = np.arange(0.8, min(0.95, args.target_iou) + 0.001, 0.05).tolist()
    noi_list, over_max_list = utils.compute_noi_metric(
        all_ious, iou_thrs=iou_thrs, max_interactions=args.n_interactions
    )

    model_name = str(logs_path.relative_to(args.logs_path)) + ':' + logs_prefix if logs_prefix else logs_path.stem
    header, table_row = utils.get_noi_table(noi_list, over_max_list, dataset_name,
                                            mean_spc, elapsed_time, args.n_interactions,
                                            model_name=model_name, mean_spo=mean_spo)

    target_iou_int = int(args.target_iou * 100)
    if target_iou_int not in [80, 85, 90]:
        noi_list, over_max_list = utils.compute_noi_metric(all_ious, iou_thrs=[args.target_iou],
                                                           max_interactions=args.n_interactions)
        table_row += f' NoI@{args.target_iou:.1%} = {noi_list[0]:.2f};'
        table_row += f' >={args.n_interactions}@{args.target_iou:.1%} = {over_max_list[0]}'

    printed = table_row
    if print_header:
        printed = header + '\n' + printed
        print_function(header)
    print_function(table_row)

    if save_ious:
        ious_path = logs_path / 'ious' / (logs_prefix if logs_prefix else '')
        ious_path.mkdir(parents=True, exist_ok=True)
        with open(ious_path / f'{dataset_name}_{args.n_interactions}.pkl', 'wb') as fp:
            pickle.dump(all_ious, fp)

    name_prefix = ''
    if logs_prefix:
        name_prefix = logs_prefix + '_'
        if not single_model_eval:
            name_prefix += f'{dataset_name}_'

    log_path = logs_path / f'{name_prefix}_{args.n_interactions}.txt'
    if log_path.exists():
        with open(log_path, 'a') as f:
            f.write(table_row + '\n')
    else:
        with open(log_path, 'w') as f:
            if print_header:
                f.write(header + '\n')
            f.write(table_row + '\n')
    return printed


def save_iou_analysis_data(
    args, dataset_name, logs_path, logs_prefix, dataset_results,
    interaction_selector=None
):
    all_ious, _, all_times = dataset_results

    name_prefix = ''
    if args.boundary_iou:
        name_prefix = 'biou_'
    if logs_prefix:
        name_prefix += logs_prefix + '_'
    name_prefix += dataset_name + '_'
    model_name = str(logs_path.relative_to(args.logs_path)) + ':' + logs_prefix if logs_prefix else logs_path.stem

    dirname = 'plots_no_zoomin' if args.no_zoomin else 'plots'
    dirname = f'product_{dirname}' if args.ui_product else dirname
    pkl_path = logs_path / f'{dirname}/{name_prefix}_{args.n_interactions}.pickle'
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with pkl_path.open('wb') as f:
        pickle.dump({
            'dataset_name': dataset_name,
            'model_name': f'{model_name}',
            'input_type': interaction_selector,
            'all_ious': all_ious,
            'all_times': all_times,
        }, f)


def get_prediction_vis_callback(logs_path, dataset_name, prob_thresh, separate_dirs):
    save_path = logs_path / 'predictions_vis' / dataset_name
    save_path.mkdir(parents=True, exist_ok=True)
    accumulated_interactions = defaultdict(list)

    def callback(image, gt_mask, prev_mask, pred_probs, sample_id, interaction_i, interactions_dict, input_type):
        ui_save_path = save_path  #  / input_type.name
        ui_save_path.mkdir(parents=True, exist_ok=True)
        prob_map = draw_probmap(pred_probs)
        # prev_mask = draw_probmap(prev_mask)
        gt_mask = draw_probmap(gt_mask)
        accumulated_interactions[sample_id].append((input_type, interactions_dict))
        pred_mask = pred_probs > prob_thresh
        vis_image_interaction = image
        for i, (itype, idict) in enumerate(accumulated_interactions[sample_id]):
            is_last_i = i == len(accumulated_interactions) - 1
            if itype == IType.point:
                vis_image_interaction = draw_with_blend_and_clicks(
                    vis_image_interaction, pred_mask, clicks_list=idict[itype],
                    enlarge_last=is_last_i
                )
            elif itype == IType.stroke or itype == IType.contour:
                vis_image_interaction = draw_with_blend_and_lines(
                    vis_image_interaction, pred_mask, interactions_list=idict[itype], alpha=0.6
                )
            else:
                raise NotImplementedError
            pred_mask = None

        im_to_save = np.concatenate((
            # prev_mask,
            vis_image_interaction, prob_map,
            gt_mask
        ), axis=1)[:, :, ::-1]
        if separate_dirs:
            sample_path = ui_save_path / f'{sample_id:04d}'
            if not sample_path.exists():
                sample_path.mkdir(parents=True)
            cv2.imwrite(str(sample_path / f'{sample_id:04d}_{interaction_i:04d}.jpg'), im_to_save)
        else:
            sample_path = ui_save_path / f'{sample_id:04d}_{interaction_i:04d}.jpg'
            cv2.imwrite(str(sample_path), im_to_save)

    return callback


if __name__ == '__main__':
    main()
