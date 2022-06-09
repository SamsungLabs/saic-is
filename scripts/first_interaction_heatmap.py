import argparse
from pathlib import Path

import seaborn as sns
from tqdm import tqdm

from isegm.data.interaction_generators.mask_transforms import get_working_area, put_result_back
from isegm.inference.utils import get_dataset
from isegm.utils.exp_imports.default import *
from isegm.utils.exp import load_config_file
from isegm.utils.vis import save_image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('output_path', type=str, help='Output path, reqiured')
    parser.add_argument('--ns', type=int, default=10, help='Number of samples to visualize')
    parser.add_argument('--ni', type=int, default=100, help='Number of interactions to visualize')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    output_images_path = Path(args.output_path)
    if not output_images_path.exists():
        output_images_path.mkdir(parents=True)

    cfg = load_config_file('config.yml', return_edict=True)
    model_cfg = edict()
    model_cfg.crop_size = (cfg.MODEL.TRAIN_SIZE, cfg.MODEL.TRAIN_SIZE)
    model_cfg.num_max_points = 1
    model_cfg.input_type = IType[cfg.MODEL.INPUT_TYPE]
    crop_size = model_cfg.crop_size

    if model_cfg.input_type == IType.point:
        generator = PointGenerator(
            max_points=model_cfg.num_max_points, prob_gamma=0.8,
            sfc_inner_k=-1, fit_normal=True, first_click_center=True
        )
    elif model_cfg.input_type == IType.contour:
        generator = ContourGenerator(
            one_component=True, convex=True,
            filled=False, width=10, shrink=False
        )
    elif model_cfg.input_type == IType.stroke:
        generator = StrokeGenerator(
            width=10, max_degree=3, one_component=True,
            axis_transform=AxisTransformType.sine
        )
    else:
        raise NotImplementedError
    trainset = get_dataset('Pilot', cfg)
    print(f'Loaded a dataset of size {len(trainset)}.')

    n_samples = min(args.ns, len(trainset)) if args.ns > 0 else len(trainset)
    for sample_i in tqdm(range(n_samples)):
        sample = trainset.get_sample(sample_i)
        image, mask = sample.image, sample.gt_mask
        imname = sample.image_name
        sample_heatmap = np.zeros(mask.shape, dtype=np.float32)
        for interaction_i in tqdm(range(args.ni), leave=False):
            if model_cfg.input_type == IType.point:
                interaction_mask = generator.generate_points_mask(mask, is_positive=True, num_points=1, radius=5)
            elif model_cfg.input_type == IType.contour:
                pad_mask, metadata = get_working_area(mask, 20, False)
                interaction_mask = generator.generate_contour_mask(pad_mask, is_positive=True)[0]
                interaction_mask = put_result_back(
                    np.zeros(mask.shape, dtype=np.float32),
                    interaction_mask, metadata, inplace=True
                )
            elif model_cfg.input_type == IType.stroke:
                interaction_mask = generator.generate_stroke_mask(mask, is_positive=True)[0]
            else:
                raise NotImplementedError
            sample_heatmap += interaction_mask / 255.
        sample_heatmap /= args.ni
        sample_heatmap = (sample_heatmap - sample_heatmap.min()) / (sample_heatmap.max() - sample_heatmap.min())

        cmap = sns.cubehelix_palette(n_colors=10, dark=0, light=0.9, as_cmap=True)
        heatmap_plot = cmap(sample_heatmap)[:, :, :-1]
        heatmap_plot[heatmap_plot == cmap(0)[:-1]] = 1

        vis_image = np.concatenate((
            np.tile(mask[:, :, None], (1, 1, 3)),
            heatmap_plot,
        ), axis=1)
        vis_image = (vis_image * 255).astype(np.uint8)

        save_image(
            output_images_path,
            Path(imname).stem, f'{sample_i:04d}',
            vis_image[:, :, ::-1],
        )
