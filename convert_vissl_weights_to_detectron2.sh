#read -p 'Model epoch: ' EPOCH
#MODEL='resnext101328d_mocov2'

EPOCH=$1
MODEL=$2
input_dir=~/project/hissl/third_party/vissl/checkpoints/${MODEL}/
output_dir=/mnt/archive/projectdata/drop/models/${MODEL}/
mkdir -p $output_dir
if [ ${#EPOCH} -eq 1 ]; then
EPOCH_OUT="00${EPOCH}"
elif [ ${#EPOCH} -eq 2 ]; then
EPOCH_OUT="0${EPOCH}"
else
EPOCH_OUT=${EPOCH}
fi

IS_FINAL_CHECKPOINT=${input_dir}model_final_checkpoint_phase${EPOCH}.torch
if [ -f "$IS_FINAL_CHECKPOINT" ]; then
    echo "$IS_FINAL_CHECKPOINT exists."
    cp $IS_FINAL_CHECKPOINT ${input_dir}model_phase${EPOCH}.torch
fi
python extra_scripts/convert_vissl_to_detectron2.py \
  --input_model_file ${input_dir}model_phase${EPOCH}.torch \
  --output_model ${output_dir}detectron2_model_phase${EPOCH_OUT}.torch \
  --weights_type torch \
  --state_dict_key_name classy_state_dict


