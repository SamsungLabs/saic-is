from isegm.utils.exp_imports.default import *

MODEL_NAME = 'cocolvis_hrnet18'


def main(cfg):
    model, model_cfg = init_model(cfg)
    train(model, cfg, model_cfg)


def init_model(cfg):
    model_cfg = edict()
    model_cfg.crop_size = (cfg.MODEL.TRAIN_SIZE, cfg.MODEL.TRAIN_SIZE)
    model_cfg.max_objects = 20
    model_cfg.input_type = IType[cfg.MODEL.INPUT_TYPE]

    model = HRNetModel(
        width=18, ocr_width=64, with_aux_output=True, use_leaky_relu=True,
        is_strided_maps_transform=True,
        use_rgb_conv=False, use_disks=True, norm_radius=5, with_prev_mask=True,
        input_type=model_cfg.input_type
    )
    model.to(cfg.device)
    if cfg.MODEL.PRETRAINED_PATH:
        model.load_pretrained_weights(cfg.MODEL.PRETRAINED_PATH)
    else:
        model.apply(initializer.XavierGluon(rnd_type='gaussian', magnitude=2.0))
        model.feature_extractor.load_pretrained_weights(cfg.IMAGENET_PRETRAINED_MODELS.HRNETV2_W18)
    return model, model_cfg


def train(model, cfg, model_cfg):
    cfg.batch_size = cfg.MODEL.BATCH_SIZE
    cfg.val_batch_size = cfg.batch_size
    crop_size = model_cfg.crop_size

    loss_cfg = edict()
    loss_cfg.instance_loss = NormalizedFocalLossSigmoid(alpha=0.5, gamma=2)
    loss_cfg.instance_loss_weight = 1.0
    loss_cfg.instance_aux_loss = SigmoidBinaryCrossEntropyLoss()
    loss_cfg.instance_aux_loss_weight = 0.4

    if cfg.MODEL.WITH_FORCE_POINTS_LOSS:
        loss_cfg.force_interactive_info_loss = MultiTypeLoss({
            IType.point: PointsForceClassBCELoss(),
            IType.contour: ContoursForceClassBCELoss(),
            IType.stroke: PointsForceClassBCELoss(),
        })
        loss_cfg.force_interactive_info_loss_weight = 0.3

    train_augmentator = Compose([
        UniformRandomResize(scale_range=(0.75, 1.40)),
        HorizontalFlip(),
        PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
        RandomCrop(*crop_size),
        RandomBrightnessContrast(brightness_limit=(-0.25, 0.25), contrast_limit=(-0.15, 0.4), p=0.75),
        RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.75)
    ], p=1.0)

    val_augmentator = Compose([
        PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
        RandomCrop(*crop_size)
    ], p=1.0)

    if model_cfg.input_type == IType.point:
        interactive_info_sampler = MultiPointSampler(
            generator=PointGenerator(
                max_points=model_cfg.max_objects, prob_gamma=0.8,
                sfc_inner_k=-1, fit_normal=True, first_click_center=True
            ),
            max_objects=model_cfg.max_objects,
            merge_objects_prob=0.15, max_num_merged_objects=2,
        )
    elif model_cfg.input_type == IType.contour:
        interactive_info_sampler = MultiContourSampler(
            generator=ContourGenerator(convex=True, one_component=True, width=10, filled=cfg.MODEL.CONTOUR_FILLED),
            max_objects=model_cfg.max_objects, positive_erode_prob=0.1,
            merge_objects_prob=0.15, max_num_merged_objects=2,
        )
    elif model_cfg.input_type == IType.stroke:
        interactive_info_sampler = MultiStrokesSampler(
            generator=StrokeGenerator(
                width=10, max_degree=3, one_component=True,
                axis_transform=AxisTransformType.sine
            ),
            max_objects=model_cfg.max_objects,
            merge_objects_prob=0.15, max_num_merged_objects=2,
        )
    else:
        raise NotImplementedError
    interactive_info_sampler = ComposeInteractionSampler(
        {model_cfg.input_type: interactive_info_sampler}, max_objects=model_cfg.max_objects,
    )

    trainset = CocoLvisDataset(
        cfg.LVIS_v1_PATH,
        split='train',
        augmentator=train_augmentator,
        min_object_area=1000,
        keep_background_prob=0.05,
        interactive_info_sampler=interactive_info_sampler,
        epoch_len=30000,
        stuff_prob=0.30,
        input_type=model_cfg.input_type
    )

    valset = CocoLvisDataset(
        cfg.LVIS_v1_PATH,
        split='val',
        augmentator=val_augmentator,
        min_object_area=1000,
        interactive_info_sampler=interactive_info_sampler,
        epoch_len=2000,
        input_type=model_cfg.input_type
    )

    num_epochs = 140
    optimizer_params = {
        'lr': 5e-4, 'betas': (0.9, 0.999), 'eps': 1e-8
    }
    lr_scheduler = partial(
        torch.optim.lr_scheduler.MultiStepLR,
        milestones=[int(0.85 * num_epochs), int(0.95 * num_epochs)], gamma=0.1
    )

    max_num_next_interactions = 3

    trainer = ISTrainer(
        model, cfg, model_cfg, loss_cfg,
        trainset, valset,
        optimizer=cfg.MODEL.OPTIMIZER_TYPE,
        optimizer_params=optimizer_params,
        lr_scheduler=lr_scheduler, lr_step_on_iter=False,
        checkpoint_interval=20,
        image_dump_interval=5000, vis_next_interactions=False,
        metrics=[AdaptiveIoU(), Accuracy()],
        max_interactive_points=model_cfg.max_objects,
        max_num_next_interactions=max_num_next_interactions,
        train_itype_selector=SingleInteractionSelector(model_cfg.input_type),
        val_itype_selector=SingleInteractionSelector(model_cfg.input_type),
    )
    trainer.run(num_epochs=num_epochs)
