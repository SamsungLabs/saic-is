import argparse
from collections import defaultdict
import os
from pathlib import Path
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sys.path.insert(0, '.')
from isegm.utils.exp import load_config_file


MEDIUM_SIZE = 12
LARGE_SIZE = 16

plt.rc('font', size=LARGE_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=LARGE_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=LARGE_SIZE)  # fontsize of the x and y labels
plt.rc('axes', titlesize=LARGE_SIZE)  # fontsize of the axes title
plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc('figure', titlesize=LARGE_SIZE)  # fontsize of the figure title
plt.rc('text', usetex=True)  # requires system TeX installation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_path', type=str,
                        help='Path(s) to experiment(s) (relative to cfg.EXPS_PATH)'
                             ' in format relative_path:checkpoint_prefix.')
    parser.add_argument('output_path', type=str,
                        help='The path to the directory to save plots into.')
    parser.add_argument('--logs-path', type=str, default='',
                        help='The path to the evaluation logs. Default path: cfg.EXPS_PATH/evaluation_logs.')
    parser.add_argument('--config-path', type=str, default='./config.yml',
                        help='The path to the config file.')
    parser.add_argument('--datasets', type=str, default='GrabCut,Berkeley,DAVIS,SBD,PascalVOC,Benchmark',
                        help='List of datasets on which the model was tested. '
                             'Datasets are separated by a comma. Possible choices: '
                             'GrabCut, Berkeley, DAVIS, SBD, PascalVOC, Benchmark')

    parser.add_argument('--n-interactions', type=str, default='30,20,10',
                        help='Number of interactions to read and display.')
    parser.add_argument('--n-columns', type=int, default=2,
                        help='Number of columns on the plot.')

    parser.add_argument('--boundary-iou', action='store_true', default=False)
    parser.add_argument('--ui-product', action='store_true', default=False)
    parser.add_argument('--with-std', action='store_true', default=False)
    parser.add_argument('--with-min-max', action='store_true', default=False)
    parser.add_argument('--with-all', action='store_true', default=False)

    args = parser.parse_args()
    cfg = load_config_file(args.config_path, return_edict=True)
    cfg.EXPS_PATH = Path(cfg.EXPS_PATH)
    if args.logs_path == '':
        args.logs_path = cfg.EXPS_PATH / 'evaluation_logs'
    else:
        args.logs_path = Path(args.logs_path)
    args.output_path = Path(args.output_path)
    return args, cfg


def get_logs_paths(args, cfg):
    logs_prefixes = []
    exp_args = args.exp_path.split(',')
    exp_paths = []
    for rel_exp_path in exp_args:
        rel_exp_path, checkpoint_prefix = rel_exp_path.split(':')
        exp_path_prefix = cfg.EXPS_PATH / rel_exp_path
        candidates = list(exp_path_prefix.parent.glob(exp_path_prefix.stem + '*'))
        exp_path = candidates[0]
        checkpoints_list = sorted((exp_path / 'checkpoints').glob(checkpoint_prefix + '*.pth'), reverse=True)
        logs_prefix = checkpoints_list[0].stem
        exp_paths.append(exp_path)
        logs_prefixes.append(logs_prefix)
    pickles_dir = 'product_plots' if args.ui_product else 'plots'
    logs_paths = [args.logs_path / epath.relative_to(cfg.EXPS_PATH) / pickles_dir for epath in exp_paths]
    n_interactions = list(map(int, args.n_interactions.split(',')))
    return logs_paths, logs_prefixes, n_interactions


def get_table(avg_iou, std_iou, dataset_name, model_name):
    table_header = (
        f'|{"Model":^50}|{"Dataset":^15}|'
        f'{"AvgIoU":^9}|{"StdIoU":^9}|'
    )
    row_width = len(table_header)
    header = '-' * row_width + '\n'
    header += table_header + '\n' + '-' * row_width
    table_row = f'|{model_name:^50}|{dataset_name:^15}|'
    table_row += f'{avg_iou:^9.2f}|'
    table_row += f'{std_iou:^9.2f}|'
    closing_line = '-' * row_width + '\n'
    return header, table_row, closing_line


def colors_mapping(model_name, last_i=None):
    label = ''
    new_i = 0
    if 'hrnet' in model_name:
        new_i = 6
        label = ' hrnet'
    elif 'segformer' in model_name:
        new_i = 7
        label = ' segformer'
    elif 'unfilled' in model_name:
        new_i = 5
        label = ' unfilled'
    elif 'accumulate' in model_name:
        new_i = 3
        label = ' accumulate'
    elif 'multi' in model_name:
        new_i = 4
    elif 'stroke' in model_name:
        new_i = 2
    elif 'point' in model_name:
        new_i = 1
    new_i = new_i if last_i is None else last_i + 1
    return new_i, label


if __name__ == '__main__':
    args, cfg = parse_args()
    logs_paths_list, logs_prefixes, n_interactions = get_logs_paths(args, cfg)
    min_interactions = min(n_interactions)

    dataset_ious = defaultdict(list)
    for lpath, lprefix in zip(logs_paths_list, logs_prefixes):
        pickles_list = []
        for filename in os.listdir(lpath):
            if lprefix not in filename:
                continue
            file_ni = int(filename.split('_')[-1][:-7])
            if file_ni not in n_interactions:
                continue
            if args.boundary_iou and filename.startswith('biou'):
                pickles_list.append(filename)
            elif not args.boundary_iou and not filename.startswith('biou'):
                pickles_list.append(filename)
        pickles_list = sorted(pickles_list)
        for pickle_name in pickles_list:
            pickle_path = os.path.join(lpath, pickle_name)
            with open(pickle_path, 'rb') as f:
                ious = pickle.load(f)
            ds_name = ious['dataset_name']
            ious_array = np.array(ious['all_ious'])
            dataset_ious[ds_name].append({
                'ious': ious_array[..., :min_interactions].mean(axis=0),
                'input_type': ious.get('input_type', 'points'),
                'model_name': lpath.parts[-2],
                'eval_path': lpath,
            })

    dataset_names = args.datasets.split(',')
    n_rows = len(dataset_names) // args.n_columns + ((len(dataset_names) % args.n_columns) > 0)
    plt.figure(figsize=(16, 6 * n_rows))
    metric_name = 'Boundary IoU' if args.boundary_iou else 'IoU'
    cmap = sns.color_palette("deep")
    model_names = set()
    print_header = True
    closing_line = None
    for dataset_i, dataset_name in enumerate(dataset_names):
        if dataset_name not in dataset_ious:
            print(f'No data for {dataset_name}')
            continue
        plt.subplot(n_rows, args.n_columns, dataset_i + 1)
        info = dataset_ious[dataset_name]
        if dataset_i % args.n_columns == 0:
            plt.ylabel(metric_name)
        if dataset_i // args.n_columns == n_rows - 1:
            plt.xlabel('interaction index')
        color_i = -1  # None
        for model_info in info:
            model_names.add(model_info['model_name'])
            label = model_info["input_type"].name
            label_fmt = '-'
            color_i, postfix = colors_mapping(model_info['model_name'], color_i)
            label = f'{label}{postfix}'
            color = cmap[color_i]
            ious = model_info['ious']
            if args.ui_product:
                mean_curve = ious.mean(0)
                plt.plot(mean_curve, label_fmt, label=label, color=color)
                if args.with_all:
                    for curve in ious:
                        plt.plot(curve, '-.', color=color, alpha=0.3)
                if args.with_std:
                    std_curve = ious.std(0)
                    plt.fill_between(
                        range(len(mean_curve)),
                        mean_curve - std_curve, mean_curve + std_curve,
                        alpha=0.3, color=color, linestyle='--', label=r'$\pm$std'
                    )
                if args.with_min_max:
                    min_curve = ious.min(0)
                    max_curve = ious.max(0)
                    plt.fill_between(
                        range(len(mean_curve)),
                        min_curve, max_curve,
                        alpha=0.3, color=color, linestyle=':', label='min-max'
                    )
                metric_average = ious.mean(1) * 100
                metric_std = metric_average.std()
                metric_average = metric_average.mean()
                header, table_row, closing_line = get_table(
                    metric_average, metric_std, dataset_name, model_info['model_name']
                )
                if print_header:
                    print(header)
                print(table_row)
                print_header = False
            else:
                plt.plot(ious, label_fmt, label=label, color=color)
            # plt.title(f'{metric_name} vs Number of interactions for {dataset_name}')
            plt.title(dataset_name)
        plt.legend(loc='lower right')
        plt.grid()
        plt.xticks(range(ious.shape[1]), range(1, ious.shape[1] + 1))
    if closing_line is not None:
        print(closing_line)
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    model_names = sorted(model_names)
    save_path = args.output_path / f'{"--".join(model_names)}--{len(dataset_names)}'
    save_path.mkdir(parents=True, exist_ok=True)
    image_name = 'iou_analysis.png' if not args.boundary_iou else 'boundary_iou_analysis.png'
    image_name = f'product_{image_name}' if args.ui_product else image_name
    image_path = save_path / image_name
    print(f'Save plot to {image_path}.')
    plt.savefig(image_path, dpi=300, bbox_inches='tight')
