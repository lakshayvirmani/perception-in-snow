_base_ = [
    '../_base_/models/upernet_r50.py', 
    '../_base_/datasets/cadcd.py',
    '../_base_/default_runtime.py', 
    '../_base_/schedules/schedule_80k.py'
]

crop_size = (512, 512)

# TODO: Should we use this or cityscapes pretrained?
pretrained = 'pretrained/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.pth'

# TODO: Check if num classes should be 1 or 2.
model = dict(
    pretrained=pretrained,
    backbone=dict(
        _delete_=True,
        type='ViTAdapter',
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        drop_path_rate=0.1,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=6,
        cffn_ratio=0.25,
        deform_ratio=1.0,
        interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],
        window_attn=[False] * 12,
        window_size=[None] * 12),
    decode_head=dict(num_classes=1, in_channels=[192, 192, 192, 192]),
    auxiliary_head=dict(num_classes=1, in_channels=192),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(341, 341))
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1024),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

optimizer = dict(_delete_=True, type='AdamW', lr=2e-5, betas=(0.9, 0.999), weight_decay=0.05,
                 constructor='LayerDecayOptimizerConstructor',
                 paramwise_cfg=dict(num_layers=24, layer_decay_rate=0.90))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

data=dict(samples_per_gpu=2,
          train=dict(pipeline=train_pipeline),
          val=dict(pipeline=test_pipeline),
          test=dict(pipeline=test_pipeline))

runner = dict(type='IterBasedRunner')

checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=1)

evaluation = dict(interval=8000, metric='mIoU', save_best='mIoU')

fp16 = dict(loss_scale=dict(init_scale=512))
