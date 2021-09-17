#We provide a config to train model using the pretext SimCLR task on the ResNet50 model. Change the DATA.TRAIN.DATA_PATHS path to the ImageNet train dataset folder path.
#for the model we need groups 32 and width per group 8
python3 tools/run_distributed_engines.py \
    config=pretrain/moco/moco_resnet50_imagenetSSL_transforms.yaml \
    config.TEST_MODEL=False \
    config.DATA.TRAIN.DATASET_NAMES=["tcga_tile_folder"] \
    config.DATA.TRAIN.DATA_SOURCES=["tile_dataset"] \
    config.DATA.TRAIN.USE_STATEFUL_DISTRIBUTED_SAMPLER=true \
    config.DATA.TRAIN.ENABLE_QUEUE_DATASET=true \
    config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=256 \
    config.OPTIMIZER.use_larc=true \
    config.OPTIMIZER.num_epochs=200 \
    config.MODEL.AMP_PARAMS.USE_AMP=true \
    config.MODEL.ACTIVATION_CHECKPOINTING.USE_ACTIVATION_CHECKPOINTING=true \
    config.MODEL.ACTIVATION_CHECKPOINTING.NUM_ACTIVATION_CHECKPOINTING_SPLITS=4 \
    config.MODEL.WEIGHTS_INIT.PARAMS_FILE="/mnt/archive/projectdata/drop/models/resnet50_trained_on_imagenet_with_moco/resnet50_trained_on_imagenet_with_moco_vissl.torch" \
    config.MODEL.WEIGHTS_INIT.APPEND_PREFIX="trunk._feature_blocks." \
    config.MODEL.WEIGHTS_INIT.STATE_DICT_KEY_NAME="model" \
    config.CHECKPOINT.DIR="./checkpoints/resnet50_moco_transforms_imagenet_init_imagenet_with_moco" \
    config.HOOKS.TENSORBOARD_SETUP.USE_TENSORBOARD=true \
    config.DISTRIBUTED.INIT_METHOD=tcp \
    config.DISTRIBUTED.NUM_NODES=1 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=4 \
    config.DISTRIBUTED.RUN_ID="localhost:51577"
