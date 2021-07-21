#!/bin/bash
read -p 'Select GPUs to use for parallel training (format example: "1,3,5"): ' GPU_IDX
SINGULARITYENV_CUDA_VISIBLE_DEVICES="${GPU_IDX}" \
singularity shell --nv \
--bind /mnt/archive/data/pathology/TCGA/gdc_manifest_all_BRCA_DX-2020-08-05/images/,/home/"$USER"/project/dlup:/dlup,/home/"$USER"/project/vissl/:/vissle,/mnt/archive/projectdata/drop/data:/data,/processing:/processing \
 /mnt/archive/containers/hissl_pytorch181_cu111_v1.sif
