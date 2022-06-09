import argparse

from tqdm import tqdm
from isegm.inference.utils import get_dataset
from isegm.utils.exp import load_config_file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default='GrabCut,Berkeley,DAVIS,SBD,PascalVOC',
                        help='List of datasets on which the model should be tested. '
                             'Datasets are separated by a comma. Possible choices: '
                             'GrabCut, Berkeley, DAVIS, SBD, PascalVOC')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg = load_config_file('config.yml', return_edict=True)
    table_header = (
        f'|{"Dataset":^15}|'
        f'{"#samples":^10}|'
        f'{"#instances":^10}|'
        f'{"Mean IpS":^10}|'
        f'{"Mean width":^15}|'
        f'{"Mean height":^15}|'
        f'{"Max width":^13}|'
        f'{"Max height":^13}|'
    )
    row_width = len(table_header)
    table_header = '-' * row_width + '\n' + table_header + '\n' + '-' * row_width
    print_header = True
    for dataset_name in args.datasets.split(','):
        dataset = get_dataset(dataset_name, cfg)
        n_samples = len(dataset)
        table_row = f'|{dataset_name:^15}|{n_samples:^10}|'
        widths = []
        heights = []
        n_instances = []
        for sample_i in tqdm(range(n_samples), leave=False):
            sample = dataset.get_sample(sample_i)
            image = sample.image
            widths.append(image.shape[1])
            heights.append(image.shape[0])
            n_instances.append(len(sample))
        table_row += f'{sum(n_instances):^10}|{sum(n_instances) / len(n_instances):^10.1f}|'
        table_row += f'{sum(widths) / len(widths):^15.3f}|{sum(heights) / len(heights):^15.3f}|'
        table_row += f'{max(widths):^13.3f}|{max(heights):^13.3f}|'
        if print_header:
            print(table_header)
            print_header = False
        print(table_row)
