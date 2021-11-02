# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from typing import Dict, List

try:
    import h5py
except ImportError:
    raise ValueError(
        "You must have h5py installed to run this script: pip install h5py."
    )

import numpy as np
import torch
import torch.distributed as dist
from classy_vision.tasks import TASK_REGISTRY, ClassyTask
from vissl.utils.io import save_file
from vissl.trainer.trainer_main import SelfSupervisionTrainer


def build_task(config):
    """Builds a ClassyTask from a config.

    This assumes a 'name' key in the config which is used to determine what
    task class to instantiate. For instance, a config `{"name": "my_task",
    "foo": "bar"}` will find a class that was registered as "my_task"
    (see :func:`register_task`) and call .from_config on it."""

    task = TASK_REGISTRY[config.TRAINER.TASK_NAME].from_config(config)
    return task


class SelfSupervisionTrainerHissl(SelfSupervisionTrainer):
    """
    See vissl.trainer.trainer_main_hissl for documentation.

    This class is a subclass of the main class, and changes the saving of
    features to include information about the tile and saves it into an h5.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def _save_extracted_features(
        features,
        targets,
        dist_rank: int,
        chunk_index: int,
        split: str,
        output_folder: str,
        meta: Dict
    ):
        #TODO Save hd5 per wsi
        #TODO if we want to do this, we need to do something smarter, otherwise a million file open and closes.
        with h5py.File(os.path.join(output_folder, f"rank{dist_rank}_chunk{chunk_index}_{split.lower()}_output.hd5"),
                       "a") as f:
            for layer_name in features.keys():
                indices = sorted(features[layer_name].keys())
                if len(indices) > 0:
                    for i in indices:
                        f[f"{meta['path'][i]}/{str(i)}/data/{layer_name}"] = \
                            features[layer_name][i]
                        f[f"{meta['path'][i]}/{str(i)}/target"] = \
                            targets[layer_name][i]
                        for item in ['x', 'y', 'h', 'w', 'mpp']:
                            f[f"{meta['path'][i]}/{str(int(meta['region_index'][i]))}/meta/{item}"] = \
                                meta[item][i]

    def _extract_split_features(
        self,
        feat_names: List[str],
        task: ClassyTask,
        split_name: str,
        output_folder: str,
    ):
        task.model.eval()
        logging.info("Model set to eval mode during feature extraction...")
        dist_rank = torch.distributed.get_rank()

        out_features, out_targets, out_meta = {}, {}, {}
        for feat_name in feat_names:
            out_features[feat_name], out_targets[feat_name] = {}, {}

        chunk_index = 0
        feature_buffer_size = 0
        while True:
            try:
                sample = next(task.data_iterator)
                assert isinstance(sample, dict)
                assert "data_idx" in sample, "Indices not passed"
                input_sample = {
                    "input": torch.cat(sample["data"]).cuda(non_blocking=True),
                    "target": torch.cat(sample["label"]).cpu().numpy(),
                    "inds": torch.cat(sample["data_idx"]).cpu().numpy(),
                    "meta": sample["meta"]
                }
                out_meta = {key: value.cpu() if isinstance(value, torch.Tensor) else value for key, value in sample["meta"][0].items()} # the dic
                logging.info(out_meta)
                with torch.no_grad():
                    features = task.model(input_sample["input"])
                    flat_features_list = self._flatten_features_list(features)
                    num_images = input_sample["inds"].shape[0]
                    print(f'\n\n num_images = {num_images}')
                    print(f'\n\n num_paths = {len(out_meta["path"])}')
                    print(f'\n\n num_region_index = {len(out_meta["region_index"])}')
                    feature_buffer_size += num_images
                    for num, feat_name in enumerate(feat_names):
                        #TODO Fix these global indices... using local at the top, global here.
                        feature = flat_features_list[num].cpu().numpy()
                        targets = input_sample["target"]
                        for idx in range(num_images):
                            index = input_sample["inds"][idx]
                            out_features[feat_name][index] = feature[idx]
                            out_targets[feat_name][index] = targets[idx].reshape(-1)

                if(
                    feature_buffer_size
                    >= self.cfg.EXTRACT_FEATURES.CHUNK_THRESHOLD
                    >= 0
                ):
                    self._save_extracted_features(
                        features=out_features,
                        targets=out_targets,
                        dist_rank=dist_rank,
                        chunk_index=chunk_index,
                        split=split_name,
                        output_folder=output_folder,
                        meta=out_meta
                    )
                    for layer_name in out_features.keys():
                        out_features[layer_name].clear()
                    chunk_index += 1
                    feature_buffer_size = 0

            except StopIteration:
                self._save_extracted_features(
                    features=out_features,
                    targets=out_targets,
                    dist_rank=dist_rank,
                    chunk_index=chunk_index,
                    split=split_name,
                    output_folder=output_folder,
                    meta=out_meta
                )
                break
