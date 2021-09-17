#We provide a config to train model using the pretext SimCLR task on the ResNet50 model. Change the DATA.TRAIN.DATA_PATHS path to the ImageNet train dataset folder path.
#for the model we need groups 32 and width per group 8
python3  tools/run_distributed_engines.py \
    config=pretrain/simclr/models/resnext101 \
    config.TEST_MODEL=False \
    config.DATA.TRAIN.DATASET_NAMES=["tcga_tile_folder"] \
    config.DATA.TRAIN.DATA_SOURCES=["tile_dataset"] \
    config.DATA.TRAIN.LABEL_TYPE="zero" \
    config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=24 \
    config.CHECKPOINT.DIR="./checkpoints" \
    config.HOOKS.TENSORBOARD_SETUP.USE_TENSORBOARD=true \
    config.DISTRIBUTED.INIT_METHOD=tcp \
    config.DISTRIBUTED.RUN_ID=auto \
    config.DISTRIBUTED.NUM_NODES=1 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=1 \

#dataset_names for the dataset is defined in the json dataset catalogue
#data source maps to the dataset class and needs to be defined in vissl/data/__init__.py. DATASET_SOURCE_MAP

#    config.DATA.COPY_DESTINATION_DIR=" /tmp/tile_data/" \
#    config.DATA.TEST.DATASET_NAMES=[''] \
#    config.DATA.TEST.DATA_SOURCES=[''] \
