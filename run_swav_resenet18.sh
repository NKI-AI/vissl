python3 tools/run_distributed_engines.py \
    config=pretrain/swav/swav_resnet18.yaml \
    config.TEST_MODEL=False \
    config.DATA.TRAIN.DATASET_NAMES=["tcga_tile_folder"] \
    config.DATA.TRAIN.DATA_SOURCES=["tile_dataset"] \
    config.DATA.TRAIN.USE_STATEFUL_DISTRIBUTED_SAMPLER=true \
    config.DATA.TRAIN.ENABLE_QUEUE_DATASET=true \
    config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=256 \
    config.OPTIMIZER.use_larc=true \
    config.MODEL.AMP_PARAMS.USE_AMP=true \
    config.MODEL.ACTIVATION_CHECKPOINTING.USE_ACTIVATION_CHECKPOINTING=false \
    config.MODEL.ACTIVATION_CHECKPOINTING.NUM_ACTIVATION_CHECKPOINTING_SPLITS=4 \
    config.CHECKPOINT.DIR="./checkpoints/swav_resnet18" \
    config.HOOKS.TENSORBOARD_SETUP.USE_TENSORBOARD=true \
    config.DISTRIBUTED.INIT_METHOD=tcp \
    config.DISTRIBUTED.NUM_NODES=1 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=2 \
    config.DISTRIBUTED.RUN_ID="localhost:50239"
