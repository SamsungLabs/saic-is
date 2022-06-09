import sys
import argparse
import multiprocessing as mp
from pathlib import Path

sys.path.insert(0, '.')
from isegm.utils.exp import load_config_file
from scripts.annotations_conversion import coco_lvis


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('dataset', choices=['coco_lvis'], help='')
    parser.add_argument('--split', nargs='+', choices=['train', 'val', 'test'], type=str, default=['train', 'val'],
                        help='')
    parser.add_argument('--n-jobs', type=int, default=10)
    parser.add_argument('--config-path', type=str, default='./config.yml',
                        help='The path to the config file.')

    args = parser.parse_args()
    cfg = load_config_file(args.config_path, return_edict=True)
    return args, cfg


def main():
    mp.set_start_method('spawn')
    args, cfg = parse_args()

    for split in args.split:
        if args.dataset == 'coco_lvis':
            coco_lvis.create_annotations(Path(cfg.LVIS_PATH), Path(cfg.COCO_PATH), dataset_split=split)


if __name__ == '__main__':
    main()
