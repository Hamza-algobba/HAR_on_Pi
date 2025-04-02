base_ = [
    '../../_base_/models/tsm_mobilenet_v2.py',
    '../../_base_/default_runtime.py'
]



# dataset settings
# dataset_type = 'VideoDataset'
# data_root = 'data/ucf_crime/videos_train/'
# data_root_val = 'data/ucf_crime/videos_val/'
# ann_file_train = 'data/ucf_crime/ucf_crime_train_list.txt'
# ann_file_val = 'data/ucf_crime/ucf_crime_val_list.txt'

dataset_type = 'VideoDataset'
data_root = 'data/ucf_crime/refined_trimmed_training_vids/'
data_root_val = 'data/ucf_crime/refined_trimmed_validation_vids/'
ann_file_train = 'data/ucf_crime/refined_train_list.txt'
ann_file_val = 'data/ucf_crime/refined_val_list.txt'

file_client_args = dict(io_backend='disk')

# class_names_file = 'data/ucf_crime/classes.txt'
class_names_file = 'data/ucf_crime/refined_classes.txt'
# ann_file_test = 'data/ucf_crime/ucf_crime_val_list.txt'
# model settings

model = dict(   
    _scope_ = 'mmaction',
    type='Recognizer2D',
    backbone=dict(
        type='MobileNetV2TSM',
        shift_div=8,
        num_segments=8,
        is_shift=True,
        pretrained='mmcls://mobilenet_v2'), 
    cls_head=dict(
        type='TSMHead',
        num_classes=2,
        in_channels=1280,
        loss_cls=dict(
            type='CrossEntropyLoss',
            class_weight=[3.0, 1.0],  # Heavier weight on 'normal' class (0)
            loss_weight=1.0),
        dropout_ratio=0.2))


# model = dict(
#     _scope_='mmaction',
#     type='Recognizer2D',
#     backbone=dict(
#         type='MobileNetV2TSM',
#         shift_div=8,
#         num_segments=8,
#         is_shift=True,
#         pretrained='mmcls://mobilenet_v2'),
#     cls_head=dict(
#         type='TSMHead',
#         num_classes=14,
#         in_channels=1280,
#         dropout_ratio=0.2))  # Increased dropout ratio from 0.25 to 0.4  (BAD, Too high, model underfits) ~Ramy 16/2



# model = dict(
#     _scope_='mmaction',
#     type='Recognizer2D',
#     backbone=dict(
#         type='ResNetTSM',         # Change to ResNetTSM
#         depth=50,                 # Use ResNet50
#         num_segments=8,          # Number of segments (TSM-specific)
#         shift_div=8,             # Shift division (TSM-specific)
#         norm_cfg=dict(type='BN', requires_grad=True),  # BatchNorm settings
#         norm_eval=False,         # If True, freezes BN during evaluation
#         partial_bn=False,        # Set True if you want partial BN
#         style='pytorch',         # PyTorch ResNet style
#         pretrained='torchvision://resnet50'  # Use torchvision ResNet50 pre-trained weights
#     ),
#     cls_head=dict(
#         type='TSMHead',
#         num_classes=14,          # Your number of classes
#         in_channels=2048,        # ResNet50 output channels (2048)
#         dropout_ratio=0.2
#     )
# )


# model = dict(
#     _scope_='mmaction',
#     type='Recognizer2D',
#     backbone=dict(
#         type='ResNetTSM',         # Change to ResNetTSM
#         depth=50,                 # Use ResNet50
#         num_segments=8,          # Number of segments (TSM-specific)
#         shift_div=8,             # Shift division (TSM-specific)
#         norm_cfg=dict(type='BN', requires_grad=True),  # BatchNorm settings
#         norm_eval=False,         # If True, freezes BN during evaluation
#         partial_bn=False,        # Set True if you want partial BN
#         style='pytorch',         # PyTorch ResNet style
#         pretrained='torchvision://resnet50'  # Use torchvision ResNet50 pre-trained weights
#     ),
#     cls_head=dict(
#         type='TSMHead',
#         num_classes=2,          # Your number of classes
#         in_channels=2048,        # ResNet50 output channels (2048)
#         dropout_ratio=0.2
#     )
# )



custom_imports = dict(
    imports=['mmaction.datasets', 
             'mmaction.datasets.transforms.loading', 
             'mmaction.datasets.transforms.processing',
             'mmaction.datasets.transforms.formatting',
             'mmaction.datasets.transforms.custom_transforms', # Import the custom transforms
             'mmaction.engine.optimizers.tsm_optim_wrapper_constructor',
             'mmaction.evaluation.metrics'],  # Ensure the datasets module from mmaction2 is imported
    allow_failed_imports=False  # Fail if the import fails
)

# class PrintFilename:
#     def __call__(self, results):
#         print(f"Training on video: {results['filename']}")
#         return results


train_pipeline = [
    # PrintFilename(),
    dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1,
        num_fixed_crops=13),
    #  dict(type='RandomRotation', degrees=5), # Added Random Rotation (15) ~Ramy 16/2 --> Reduced to 7 ~Ramy 17/2
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    
    dict(
        type='MultiScaleCrop', 
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1,
        num_fixed_crops=13),
    
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]

test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]

# data = dict(
#     samples_per_gpu =32,
#     workers_per_gpu = 2,
#     train = dict(
#         type = dataset_type,
#         data_prefix = data_root,
#         ann_file = ann_file_train,
#         classes = class_names_file,
#         pipeline = train_pipeline
#     ),
#     val = dict(
#         type = dataset_type,
#         data_prefix = data_root_val,
#         ann_file = ann_file_val,
#         classes = class_names_file,
#         pipeline = val_pipeline
#     ),
#     test = dict(
#         type = dataset_type,
#         data_prefix = data_root_val,
#         ann_file = ann_file_test,
#         classes = class_names_file,
#         pipeline = test_pipeline
#     )
# )

train_dataloader = dict(
    batch_size=8, #changed from 16 to 32 (out of memory max: 20) ~Hambozo 22/2
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=8, #changed from 16 to 8 ( ~Hambozo 1/3)
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val),
        pipeline=val_pipeline,
        test_mode=True))
test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val),
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

default_hooks = dict(checkpoint=dict(type = 'CheckpointHook',interval=3, max_keep_ckpts=3))

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=50,  # change from 100 to 50
    val_begin=1,
    val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')


param_scheduler = [                   # Default scheduler
    dict(
        type='MultiStepLR',
        begin=0,
        end=100,  # change from 100 to 50
        by_epoch=True,
        milestones=[20, 40, 60, 80],  
        gamma=0.1)
]

# param_scheduler = [
#     dict(type='LinearLR', start_factor=0.001, by_epoch=True, begin=0, end=5),   # Different Param Scheduler ~Ramy 16/2
#     dict(type='MultiStepLR', milestones=[20, 40, 60, 80], gamma=0.1)
# ]

# param_scheduler = [
#     dict(type='CosineAnnealingLR', T_max=100, eta_min=1e-5, by_epoch=True) # Different Param Scheduler ~Ramy 17/2
# ]

optim_wrapper = dict(
    constructor='TSMOptimWrapperConstructor',
    paramwise_cfg=dict(fc_lr5=True),
    # optimizer=dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=0.0001), # Default optimizer
    optimizer= dict(type='AdamW', lr=0.0005, weight_decay=0.0001), # Changed from SGD to Adam ~Ramy 17/2
    clip_grad=dict(max_norm=20, norm_type=2))

auto_scale_lr = dict(enable=True, base_batch_size=128)
# use the pre-trained model for the whole TSN network
#load_from = '/home/g6/Desktop/Thesis/CV_models/mmaction/mmaction2/tsm_imagenet-pretrained-mobilenetv2_8xb16-1x1x8-100e_kinetics400-rgb_20230414-401127fd.pth'

load_from = 'tsm_imagenet-pretrained-r50_8xb16-dense-1x1x8-50e_kinetics400-rgb_20220831-f55d3c2b.pth'