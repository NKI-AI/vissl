#We provide a config to train model using the pretext SimCLR task on the ResNet50 model. Change the DATA.TRAIN.DATA_PATHS path to the ImageNet train dataset folder path.
#for the model we need groups 32 and width per group 8
python3  tools/run_distributed_engines.py \
    hydra.verbose=true \
    config=pretrain/simclr/models/resnext101 \
    config.DATA.TRAIN.DATASET_NAMES=['tcga_tile_folder'] \
    config.CHECKPOINT.DIR="./checkpoints" \
    config.HOOKS.TENSORBOARD_SETUP.USE_TENSORBOARD=true \
     config.DISTRIBUTED.INIT_METHOD=tcp \
    config.DISTRIBUTED.RUN_ID=auto \
    config.DISTRIBUTED.NUM_NODES=1 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=1 \

