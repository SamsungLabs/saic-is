from collections import defaultdict
from copy import deepcopy
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from isegm.data.interaction_type import IType, RandomInteractionSelector, InteractionSelector
from isegm.data.next_input_getters import get_next_points, get_next_contour_mask, get_next_strokes
from isegm.engine.optimizer import get_optimizer
from isegm.utils.distributed import get_dp_wrapper, get_sampler, reduce_loss_dict
from isegm.utils.log import logger, TqdmToLogger, SummaryWriterAvg
from isegm.utils.misc import save_checkpoint
from isegm.utils.serialization import get_config_repr
from isegm.utils.vis import draw_probmap, draw_points, save_image, blend_with_interaction_mask


class ISTrainer(object):
    def __init__(
        self, model, cfg, model_cfg, loss_cfg, trainset, valset,
        optimizer='adam', optimizer_params=None,
        image_dump_interval=200, checkpoint_interval=10, tb_dump_period=25,
        max_interactive_points=0, max_num_next_interactions=0, vis_next_interactions=False,
        lr_scheduler=None, lr_step_on_iter=False,
        metrics=None, additional_val_metrics=None,
        prev_mask_drop_prob=0.0, input_drop_prob=0.9,
        train_itype_selector=RandomInteractionSelector(), val_itype_selector=InteractionSelector(),
        accumulate_interactions=False,
    ):
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.max_interactive_points = max_interactive_points
        self.loss_cfg = loss_cfg
        self.val_loss_cfg = deepcopy(loss_cfg)
        self.tb_dump_period = tb_dump_period
        self.max_num_next_interactions = max_num_next_interactions
        self.vis_next_interactions = vis_next_interactions
        self.train_iselector = train_itype_selector
        self.val_iselector = val_itype_selector
        self.accumulate_interactions = accumulate_interactions

        self.prev_mask_drop_prob = prev_mask_drop_prob
        self.input_drop_prob = input_drop_prob
        self.optimizer_type = optimizer
        self.lr_step_on_iter = lr_step_on_iter

        if cfg.distributed:
            cfg.batch_size //= cfg.ngpus
            cfg.val_batch_size //= cfg.ngpus

        if metrics is None:
            metrics = []
        self.train_metrics = metrics
        self.val_metrics = deepcopy(metrics)
        if additional_val_metrics is not None:
            self.val_metrics.extend(additional_val_metrics)

        self.checkpoint_interval = checkpoint_interval
        self.image_dump_interval = image_dump_interval
        self.task_prefix = ''
        self.sw = None

        self.trainset = trainset
        self.train_generator = trainset.interactive_info_sampler.generator
        self.valset = valset

        logger.info(f'Dataset of {trainset.get_samples_number()} samples was loaded for training.')
        logger.info(f'Dataset of {valset.get_samples_number()} samples was loaded for validation.')

        self.train_data = DataLoader(
            trainset, cfg.batch_size,
            sampler=get_sampler(trainset, shuffle=True, distributed=cfg.distributed),
            drop_last=True, pin_memory=True,
            num_workers=cfg.workers
        )

        self.val_data = DataLoader(
            valset, cfg.val_batch_size,
            sampler=get_sampler(valset, shuffle=False, distributed=cfg.distributed),
            drop_last=True, pin_memory=True,
            num_workers=cfg.workers
        )

        self.optim = get_optimizer(model, optimizer, optimizer_params)
        model = self._load_weights(model)

        if cfg.multi_gpu:
            model = get_dp_wrapper(cfg.distributed)(model, device_ids=cfg.gpu_ids,
                                                    output_device=cfg.gpu_ids[0])

        if self.is_master:
            logger.info(model)
            logger.info(get_config_repr(model._config))

        self.device = cfg.device
        self.net = model.to(self.device)
        self.lr = optimizer_params['lr']

        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler(optimizer=self.optim)
            if cfg.start_epoch > 0:
                for _ in range(cfg.start_epoch):
                    self.lr_scheduler.step()

        self.tqdm_out = TqdmToLogger(logger, level=logging.INFO)

    def run(self, num_epochs, start_epoch=None, validation=True):
        if start_epoch is None:
            start_epoch = self.cfg.start_epoch

        logger.info(f'Starting Epoch: {start_epoch}')
        logger.info(f'Total Epochs: {num_epochs}')
        for epoch in range(start_epoch, num_epochs):
            self.training(epoch)
            if validation:
                self.validation(epoch)

    def training(self, epoch):
        if self.sw is None and self.is_master:
            self.sw = SummaryWriterAvg(log_dir=str(self.cfg.LOGS_PATH),
                                       flush_secs=10, dump_period=self.tb_dump_period)

        if self.cfg.distributed:
            self.train_data.sampler.set_epoch(epoch)

        log_prefix = 'Train' + self.task_prefix.capitalize()
        tbar = tqdm(self.train_data, file=self.tqdm_out, ncols=120)\
            if self.is_master else self.train_data

        for metric in self.train_metrics:
            metric.reset_epoch_stats()

        self.net.train()
        train_loss = 0.0
        for i, batch_data in enumerate(tbar):
            global_step = epoch * len(self.train_data) + i

            loss, losses_logging, splitted_batch_data, outputs = self.batch_forward(batch_data)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            losses_logging['overall'] = loss
            reduce_loss_dict(losses_logging)

            train_loss += losses_logging['overall'].item()

            if self.is_master:
                for loss_name, loss_value in losses_logging.items():
                    self.sw.add_scalar(tag=f'{log_prefix}Losses/{loss_name}',
                                       value=loss_value.item(),
                                       global_step=global_step)

                for k, v in self.loss_cfg.items():
                    if '_loss' in k and hasattr(v, 'log_states') and self.loss_cfg.get(k + '_weight', 0.0) > 0:
                        v.log_states(self.sw, f'{log_prefix}Losses/{k}', global_step)

                if self.image_dump_interval > 0 and global_step % self.image_dump_interval == 0:
                    self.save_visualization(splitted_batch_data, outputs, global_step, prefix='train')

                self.sw.add_scalar(
                    tag=f'{log_prefix}States/learning_rate',
                    value=self.lr if not hasattr(self, 'lr_scheduler') else max(self.lr_scheduler.get_last_lr()),
                    global_step=global_step
                )

                metric_str = ""
                for metric in self.train_metrics:
                    metric.log_states(self.sw, f'{log_prefix}Metrics/{metric.name}', global_step)
                    metric_str += f' Metrics/{metric.name}={metric.get_epoch_value():.4f}'
                tbar.set_description(f'Epoch {epoch}, training loss {train_loss/(i+1):.4f}')
            if self.lr_step_on_iter:
                self.lr_scheduler.step()

        if self.is_master:
            for metric in self.train_metrics:
                self.sw.add_scalar(tag=f'{log_prefix}Metrics/{metric.name}',
                                   value=metric.get_epoch_value(),
                                   global_step=epoch, disable_avg=True)

            save_checkpoint(self.net, self.cfg.CHECKPOINTS_PATH, prefix=self.task_prefix,
                            epoch=None, multi_gpu=self.cfg.multi_gpu)

            if isinstance(self.checkpoint_interval, (list, tuple)):
                checkpoint_interval = [x for x in self.checkpoint_interval if x[0] <= epoch][-1][1]
            else:
                checkpoint_interval = self.checkpoint_interval

            if epoch % checkpoint_interval == 0:
                save_checkpoint(self.net, self.cfg.CHECKPOINTS_PATH, prefix=self.task_prefix,
                                epoch=epoch, multi_gpu=self.cfg.multi_gpu)

        if hasattr(self, 'lr_scheduler') and not self.lr_step_on_iter:
            self.lr_scheduler.step()

    def validation(self, epoch):
        if self.sw is None and self.is_master:
            self.sw = SummaryWriterAvg(log_dir=str(self.cfg.LOGS_PATH),
                                       flush_secs=10, dump_period=self.tb_dump_period)

        log_prefix = 'Val' + self.task_prefix.capitalize()
        tbar = tqdm(self.val_data, file=self.tqdm_out, ncols=100) if self.is_master else self.val_data

        for metric in self.val_metrics:
            metric.reset_epoch_stats()

        val_loss = 0
        losses_logging = defaultdict(list)

        self.net.eval()
        for i, batch_data in enumerate(tbar):
            global_step = epoch * len(self.val_data) + i
            loss, batch_losses_logging, splitted_batch_data, outputs = self.batch_forward(batch_data, validation=True)

            batch_losses_logging['overall'] = loss
            reduce_loss_dict(batch_losses_logging)
            for loss_name, loss_value in batch_losses_logging.items():
                losses_logging[loss_name].append(loss_value.item())

            val_loss += batch_losses_logging['overall'].item()

            if self.is_master:
                metric_str = ""
                for metric in self.val_metrics:
                    metric.log_states(self.sw, f'{log_prefix}Metrics/{metric.name}', global_step)
                    metric_str += f' Metrics/{metric.name}={metric.get_epoch_value():.4f}'
                tbar.set_description(
                    f'Epoch {epoch}, validation loss: {val_loss/(i + 1):.4f}'
                    f' {metric_str.replace("Metrics/", "")}'
                )

        if self.is_master:
            for loss_name, loss_values in losses_logging.items():
                self.sw.add_scalar(tag=f'{log_prefix}Losses/{loss_name}', value=np.array(loss_values).mean(),
                                   global_step=epoch, disable_avg=True)

            for metric in self.val_metrics:
                self.sw.add_scalar(tag=f'{log_prefix}Metrics/{metric.name}', value=metric.get_epoch_value(),
                                   global_step=epoch, disable_avg=True)

    def batch_forward(self, batch_data, validation=False):
        metrics = self.val_metrics if validation else self.train_metrics
        interaction_selector = self.val_iselector if validation else self.train_iselector
        losses_logging = dict()

        with torch.set_grad_enabled(not validation):
            batch_data = {k: v.to(self.device) for k, v in batch_data.items()}
            image, gt_mask = batch_data['images'].clone(), batch_data['instances'].clone()
            input_type = interaction_selector.select_itype(0, gt_mask)
            interactive_info = {}
            for itype in interaction_selector.itypes:
                key = f'{itype.name}_interactive_info'
                if input_type == itype:
                    interactive_info[key] = batch_data[key].clone()
                elif np.random.rand() < self.input_drop_prob:
                    if itype == IType.point:
                        interactive_info[key] = torch.zeros_like(batch_data[key]) - 1
                    else:
                        interactive_info[key] = torch.zeros_like(batch_data[key])
                else:
                    interactive_info[key] = batch_data[key].clone()
            orig_interactive_info = batch_data[f'{input_type.name}_interactive_info'].clone()
            orig_input_type = input_type

            num_iters = random.randint(0, self.max_num_next_interactions)
            prev_output, interactive_info, interactive_gt, all_outputs, input_type = self.make_eval_steps(
                num_steps=num_iters,
                image=image, gt_mask=gt_mask, interactive_info=interactive_info,
                init_input_type=input_type, itype_selector=interaction_selector, validation=validation
            )
            batch_data.update(interactive_info)
            batch_data['input_type'] = input_type.name

            net_input = torch.cat((image, prev_output), dim=1) if self.net.with_prev_mask else image
            output = self.net(net_input, interactive_info, input_type=input_type)
            output['prev_outputs'] = prev_output
            output['orig_interactive_info'] = orig_interactive_info
            output['orig_input_type'] = orig_input_type.name
            output['all_outputs'] = all_outputs

            loss = 0.0
            loss = self.add_loss('instance_loss', loss, losses_logging, validation,
                                 lambda: ((output['instances'], batch_data['instances']), {}))
            loss = self.add_loss('instance_aux_loss', loss, losses_logging, validation,
                                 lambda: ((output['instances_aux'], batch_data['instances']), {}))
            loss = self.add_loss('force_interactive_info_loss', loss, losses_logging, validation,
                                 lambda: ((output['instances'], interactive_gt), {'input_type': input_type}))

            if self.is_master:
                with torch.no_grad():
                    for m in metrics:
                        try:
                            m.update(*(output.get(x) for x in m.pred_outputs),
                                     *(batch_data[x] for x in m.gt_outputs))
                        except RuntimeError as exc:
                            logger.info(f'Exception in metric {m}: {str(exc)}')
                            continue

        return loss, losses_logging, batch_data, output

    def make_eval_steps(
        self, num_steps,
        image, gt_mask, interactive_info,
        init_input_type, itype_selector, validation
    ):
        prev_output = torch.zeros_like(image, dtype=torch.float32)[:, :1, :, :]
        input_type = init_input_type
        last_click_indx = None
        interactive_gt = None
        with torch.no_grad():
            all_outputs = []
            if not validation:
                self.net.eval()
            eval_model = self.net
            for interaction_i in range(num_steps):
                last_click_indx = interaction_i
                net_input = torch.cat((image, prev_output), dim=1) if self.net.with_prev_mask else image
                prev_output = torch.sigmoid(eval_model(
                    net_input, interactive_info,
                    input_type=input_type,
                )['instances'])
                if self.vis_next_interactions:
                    all_outputs.append({
                        'interactive_info': interactive_info[f'{input_type.name}_interactive_info'].clone(),
                        'output': prev_output.clone(),
                        'prev_output': net_input[:, -1].clone(),
                        'input_type': input_type.name,
                    })

                error_map = (gt_mask > 0) & (prev_output < 0.5)
                input_type = itype_selector.select_itype(interaction_i + 1, error_map)
                if input_type == IType.point:
                    iinfo = get_next_points(
                        prev_output, gt_mask, interactive_info[f'{input_type.name}_interactive_info'],
                        self.train_generator[input_type], interaction_i + 1,
                    )
                    interactive_gt = iinfo.clone()
                elif input_type == IType.contour:
                    iinfo, interactive_gt = get_next_contour_mask(
                        prev_output, gt_mask, self.train_generator[input_type]
                    )
                    if self.accumulate_interactions:
                        iinfo = (interactive_info[f'{input_type.name}_interactive_info'] > 0) | (iinfo > 0)
                elif input_type == IType.stroke:
                    iinfo, interactive_gt = get_next_strokes(
                        prev_output, gt_mask, self.train_generator[input_type]
                    )
                    if self.accumulate_interactions:
                        iinfo = (interactive_info[f'{input_type.name}_interactive_info'] > 0) | (iinfo > 0)
                else:
                    raise NotImplementedError
                interactive_info[f'{input_type.name}_interactive_info'] = iinfo

            if not validation:
                self.net.train()
            if self.net.with_prev_mask and self.prev_mask_drop_prob > 0 and last_click_indx is not None:
                zero_mask = np.random.random(size=prev_output.size(0)) < self.prev_mask_drop_prob
                prev_output[zero_mask] = torch.zeros_like(prev_output[zero_mask])
        return prev_output, interactive_info, interactive_gt, all_outputs, input_type

    def add_loss(self, loss_name, total_loss, losses_logging, validation, lambda_loss_inputs):
        loss_cfg = self.loss_cfg if not validation else self.val_loss_cfg
        loss_weight = loss_cfg.get(loss_name + '_weight', 0.0)
        if loss_weight > 0.0:
            loss_criterion = loss_cfg.get(loss_name)
            try:
                loss_args, loss_kwargs = lambda_loss_inputs()
                loss = loss_criterion(*loss_args, **loss_kwargs)
            except RuntimeError as exc:
                logger.info(f'Exception in loss {loss_criterion}: {str(exc)}')
                loss = torch.tensor([0.0], device=self.device)
            loss = torch.mean(loss)
            losses_logging[loss_name] = loss
            loss = loss_weight * loss
            total_loss = total_loss + loss

        return total_loss

    def save_visualization(self, splitted_batch_data, outputs, global_step, prefix):
        output_images_path = self.cfg.VIS_PATH / prefix
        if self.task_prefix:
            output_images_path /= self.task_prefix
        if not output_images_path.exists():
            output_images_path.mkdir(parents=True)
        image_name_prefix = f'{global_step:06d}'

        image_blob = splitted_batch_data['images'][0]
        input_type = splitted_batch_data['input_type']
        interactions = extract_and_detach(splitted_batch_data[f'{input_type}_interactive_info'])
        gt_mask = extract_and_detach(splitted_batch_data['instances'], squeeze=True)
        predicted_mask = extract_and_detach(torch.sigmoid(outputs['instances']), squeeze=True)
        prev_output = extract_and_detach(outputs['prev_outputs'], squeeze=True)
        orig_input_type = outputs['orig_input_type']
        orig_interactions = extract_and_detach(outputs['orig_interactive_info'])

        image = image_blob.cpu().numpy() * 255
        image = image.transpose((1, 2, 0))
        gt_mask[gt_mask < 0] = 0.25
        gt_mask = draw_probmap(gt_mask)
        predicted_mask = draw_probmap(predicted_mask)
        prev_output = draw_probmap(prev_output)
        if input_type == IType.point.name:
            image_with_interaction = self.visualize_last_points(image, interactions)
        else:
            interactions = interactions.transpose((1, 2, 0))
            image_with_interaction = blend_with_interaction_mask(image, interactions, alpha=0.5)
        if orig_input_type == IType.point.name:
            image_with_orig_interaction = self.visualize_last_points(image, orig_interactions)
        else:
            orig_interactions = orig_interactions.transpose((1, 2, 0))
            image_with_orig_interaction = blend_with_interaction_mask(image, orig_interactions, alpha=0.5)
        viz_image = np.hstack((
            image_with_orig_interaction, prev_output,
            image_with_interaction, predicted_mask,
            gt_mask
        )).astype(np.uint8)
        save_image(output_images_path, image_name_prefix, 'instance_segmentation', viz_image[:, :, ::-1])

        if self.vis_next_interactions and len(outputs['all_outputs']) > 0:
            viz_last_interaction = np.hstack((prev_output, image_with_interaction, predicted_mask))
            all_itypes = [v['input_type'] for v in outputs['all_outputs']]
            all_prev = [extract_and_detach(v['prev_output']) for v in outputs['all_outputs']]
            all_info = [extract_and_detach(v['interactive_info']) for v in outputs['all_outputs']]
            all_output = [extract_and_detach(v['output'], squeeze=True) for v in outputs['all_outputs']]
            all_prev = [draw_probmap(v) for v in all_prev]
            all_output = [draw_probmap(v) for v in all_output]
            all_with_interaction = []
            for itype, info in zip(all_itypes, all_info):
                if itype == IType.point.name:
                    image_with_interaction = self.visualize_last_points(image, info)
                else:
                    info = info.transpose((1, 2, 0))
                    image_with_interaction = blend_with_interaction_mask(image, info, alpha=0.5)
                all_with_interaction.append(image_with_interaction)
            all_with_interaction = np.vstack(all_with_interaction)
            all_output = np.vstack(all_output)
            all_prev = np.vstack(all_prev)
            viz_image = np.hstack((all_prev, all_with_interaction, all_output))
            viz_image = np.vstack((viz_image, viz_last_interaction))
            viz_image = viz_image.astype(np.uint8)
            save_image(output_images_path, image_name_prefix, 'instance_segmentation_steps', viz_image[:, :, ::-1])

    def _load_weights(self, net):
        if self.cfg.weights is not None:
            if os.path.isfile(self.cfg.weights):
                load_weights(net, self.cfg.weights)
                self.cfg.weights = None
            else:
                raise RuntimeError(f"=> no checkpoint found at '{self.cfg.weights}'")
        elif self.cfg.resume_exp is not None:
            checkpoints = list(self.cfg.CHECKPOINTS_PATH.glob(f'{self.cfg.resume_prefix}*.pth'))
            assert len(checkpoints) == 1

            checkpoint_path = checkpoints[0]
            logger.info(f'Load checkpoint from path: {checkpoint_path}')
            load_weights(net, str(checkpoint_path))
        return net

    @property
    def is_master(self):
        return self.cfg.local_rank == 0

    def visualize_last_points(self, image, points):
        image_with_points = draw_points(
            image,
            # points[self.max_interactive_points - self.max_num_next_interactions:self.max_interactive_points],
            points[:self.max_interactive_points],
            (0, 255, 0), radius=4,
        )
        image_with_points = draw_points(
            image_with_points,
            # points[-self.max_num_next_interactions:],
            points[self.max_interactive_points:],
            (255, 0, 0), radius=4,
        )
        return image_with_points


def load_weights(model, path_to_weights):
    current_state_dict = model.state_dict()
    new_state_dict = torch.load(path_to_weights, map_location='cpu')['state_dict']
    current_state_dict.update(new_state_dict)
    model.load_state_dict(current_state_dict)


def extract_and_detach(tensor, take_index=0, squeeze=False):
    result = tensor.detach().cpu().numpy()[take_index]
    if squeeze:
        result = np.squeeze(result, axis=0)
    return result
