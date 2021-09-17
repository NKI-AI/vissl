#We provide a config to train model using the pretext SimCLR task on the ResNet50 model. Change the DATA.TRAIN.DATA_PATHS path to the ImageNet train dataset folder path.
#for the model we need groups 32 and width per group 8
python3 tools/run_distributed_engines.py \
    config=pretrain/moco/moco_resnext101 \
    config.TEST_MODEL=False \
    config.DATA.TRAIN.DATASET_NAMES=["tcga_tile_folder"] \
    config.DATA.TRAIN.DATA_SOURCES=["tile_dataset"] \
    config.DATA.TRAIN.USE_STATEFUL_DISTRIBUTED_SAMPLER=true \
    config.DATA.TRAIN.ENABLE_QUEUE_DATASET=true \
    config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=256 \
    config.OPTIMIZER.use_larc=true \
    config.MODEL.TRUNK.RESNETS.WIDTH_PER_GROUP=8 \
    config.MODEL.AMP_PARAMS.USE_AMP=true \
    config.MODEL.ACTIVATION_CHECKPOINTING.USE_ACTIVATION_CHECKPOINTING=true \
    config.MODEL.ACTIVATION_CHECKPOINTING.NUM_ACTIVATION_CHECKPOINTING_SPLITS=4 \
    config.CHECKPOINT.DIR="./checkpoints/resnext101328d_mocov2_testamp" \
    config.HOOKS.TENSORBOARD_SETUP.USE_TENSORBOARD=true \
    config.DISTRIBUTED.INIT_METHOD=tcp \
    config.DISTRIBUTED.NUM_NODES=1 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=4 \
    config.DISTRIBUTED.RUN_ID="localhost:54245"


# foreign tcp        0      9 127.0.0.1:56459         127.0.0.1:46424         ESTABLISHED
#35065
#54245  56459
#    config.DATA.TRAIN.DATA_SOURCES=[disk_folder] \
#    config.DATA.TRAIN.DATA_PATHS=["/path/to/my/imagenet/folder/train"] \

#    config.DATA.TRAIN.DATASET_NAMES=["tcga_tile_folder"] \
#    config=test/integration_test/quick_simclr_imagefolder \
# config.DISTRIBUTED.RUN_ID="localhost:56459"tcp://localhost:56459,

##./vissl/configs/config/
   # config.OPTIMIZER.use_larc=true \

#    config.MODEL.AMP_PARAMS.USE_AMP=true \

#    config.MODEL.ACTIVATION_CHECKPOINTING.USE_ACTIVATION_CHECKPOINTING=true \
#    config.MODEL.ACTIVATION_CHECKPOINTING.NUM_ACTIVATION_CHECKPOINTING_SPLITS=4 \
#    config.OPTIMIZER.use_larc=true \
#    config.OPTIMIZER.use_zero=true \

