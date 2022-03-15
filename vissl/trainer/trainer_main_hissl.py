# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from typing import Dict, List
from pathlib import Path

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
    def _original_save_extracted_features(
        features,
        targets,
        dist_rank: int,
        chunk_index: int,
        split: str,
        output_folder: str,
    ):
        output = {}
        for layer_name in features.keys():
            indices = sorted(features[layer_name].keys())
            if len(indices) > 0:
                output[layer_name] = {
                    "inds": np.array(indices),
                    "features": np.array([features[layer_name][i] for i in indices]),
                    "targets": np.array([targets[layer_name][i] for i in indices]),
                }

        for layer_name, layer_features in output.items():
            out_feat_file = os.path.join(
                output_folder,
                f"rank{dist_rank}_chunk{chunk_index}_{split.lower()}_{layer_name}_features.npy",
            )
            out_target_file = os.path.join(
                output_folder,
                f"rank{dist_rank}_chunk{chunk_index}_{split.lower()}_{layer_name}_targets.npy",
            )
            out_inds_file = os.path.join(
                output_folder,
                f"rank{dist_rank}_chunk{chunk_index}_{split.lower()}_{layer_name}_inds.npy",
            )
            save_file(layer_features["features"], out_feat_file)
            save_file(layer_features["targets"], out_target_file)
            save_file(layer_features["inds"], out_inds_file)


    @staticmethod
    def _save_dlup_wsi_features(
        features,
        targets,
        dist_rank: int,
        chunk_index: int,
        split: str,
        output_folder: str,
        meta: Dict,
    ):
        with h5py.File(os.path.join(output_folder, f"rank{dist_rank}_{split.lower()}_output.hd5"),
                       "a") as f:
            for layer_name in features.keys():
                indices = sorted(features[layer_name].keys())
                if len(indices) > 0:
                    for i in indices:
                        f[f"{meta[layer_name][i]['path']}/{str(i)}/data/{layer_name}"] = \
                            features[layer_name][i]
                        f[f"{meta[layer_name][i]['path']}/{str(i)}/target"] = \
                            targets[layer_name][i]
                        for item in ['x', 'y', 'h', 'w', 'mpp', 'region_index']:
                            f[f"{meta[layer_name][i]['path']}/{str(i)}/meta/{item}"] = \
                                meta[layer_name][i][item]

    @staticmethod
    def _save_kather_msi_features(
        features,
        targets,
        dist_rank: int,
        chunk_index: int,
        split: str,
        output_folder: str,
        meta: Dict,
    ):
        # """
        # Note that this cannot currently be run with more than 1 GPU. Please set num_GPUs to 1.
        # """
        with h5py.File(os.path.join(output_folder, f"rank{dist_rank}_{split.lower()}_output.hd5"),
                       "a") as f:
            for layer_name in features.keys():
                indices = sorted(features[layer_name].keys())
                if len(indices) > 0:
                    for i in indices:
                        f[f"{meta[layer_name][i]['path']}/data/{layer_name}"] = \
                            features[layer_name][i]
                        f[f"{meta[layer_name][i]['path']}/target"] = \
                            targets[layer_name][i]
                        f[f"{meta[layer_name][i]['path']}/meta/vissl_id"] = \
                            i

                        for item in ['case_id', 'slide_id']:
                            f[f"{meta[layer_name][i]['path']}/meta/{item}"] = \
                                meta[layer_name][i][item]


    @staticmethod
    def _save_extracted_features(
        self,
        features,
        targets,
        dist_rank: int,
        chunk_index: int,
        split: str,
        output_folder: str,
        meta: Dict,
        dataset_name: str
    ):
        #TODO Save hd5 per wsi. Currently done retrospectively in DLUP-LIGHTNING-MIL
        if dataset_name == 'dlup_wsi':
            self._save_dlup_wsi_features(features=features,
                                         targets=targets,
                                         dist_rank=dist_rank,
                                         chunk_index=chunk_index,
                                         split=split,
                                         output_folder=output_folder,
                                         meta=meta)
        elif dataset_name == 'kather_msi_dataset':
            self._save_kather_msi_features(features=features,
                                           targets=targets,
                                           dist_rank=dist_rank,
                                           chunk_index=chunk_index,
                                           split=split,
                                           output_folder=output_folder,
                                           meta=meta)
        else:
            self._original_save_extracted_features(
                features=features,
                targets=targets,
                dist_rank=dist_rank,
                chunk_index=chunk_index,
                split=split,
                output_folder=output_folder
            )


    def _extract_split_features(
        self,
        feat_names: List[str],
        task: ClassyTask,
        split_name: str,
        output_folder: str,
    ):
        task.model.eval()

        data_sources = self.cfg.DATA[split_name].DATA_SOURCES
        if len(data_sources) > 1:
            logging.error("Hissl can only extract features when a single data source type is given")
            raise NotImplementedError
        data_source = data_sources[0]

        logging.info("Model set to eval mode during feature extraction...")

        dist_rank = torch.distributed.get_rank()

        out_features, out_targets, out_meta = {}, {}, {}
        for feat_name in feat_names:
            out_features[feat_name], out_targets[feat_name], out_meta[feat_name] = {}, {}, {}

        chunk_index = 0
        feature_buffer_size = 0

        while True:
            try:
                logging.info(f"Batch #{chunk_index} being saved...")
                sample = next(task.data_iterator)
                assert isinstance(sample, dict)
                assert "data_idx" in sample, "Indices not passed"
                input_sample = {
                    "input": torch.cat(sample["data"]).cuda(non_blocking=True),
                    "target": torch.cat(sample["label"]).cpu().numpy(),
                    "inds": torch.cat(sample["data_idx"]).cpu().numpy(),

                }
                if "meta" in sample.keys():
                    input_meta = sample["meta"][0]

                with torch.no_grad():
                    features = task.model(input_sample["input"])
                    flat_features_list = self._flatten_features_list(features)
                    num_images = input_sample["inds"].shape[0]
                    feature_buffer_size += num_images
                    for feat_num, feat_name in enumerate(feat_names):
                        #TODO Fix these global indices. Local and global features are confusing.
                        feature = flat_features_list[feat_num].cpu().numpy()
                        targets = input_sample["target"]
                        for img_idx in range(num_images):
                            dataset_index = input_sample["inds"][img_idx]
                            # The dataset index is set as a key in out_features and out_targets and out_meta
                            if 'meta' in sample.keys():
                                # out_meta = {key: value.cpu() if isinstance(value, torch.Tensor) else value for
                                #             key, value in sample["meta"][0].items()}  # the dic
                                out_meta[feat_name][dataset_index] = {key: value.cpu()[img_idx] if isinstance(value, torch.Tensor) else value[img_idx] for key, value in input_meta.items()}
                            out_features[feat_name][dataset_index] = feature[img_idx]
                            out_targets[feat_name][dataset_index] = targets[img_idx].reshape(-1)

                if(
                    feature_buffer_size
                    >= self.cfg.EXTRACT_FEATURES.CHUNK_THRESHOLD
                    >= 0
                ):
                    self._save_extracted_features(
                        self=self,
                        features=out_features,
                        targets=out_targets,
                        dist_rank=dist_rank,
                        chunk_index=chunk_index,
                        split=split_name,
                        output_folder=output_folder,
                        meta=out_meta,
                        dataset_name=data_source
                    )
                    for layer_name in out_features.keys():
                        out_features[layer_name].clear()
                    chunk_index += 1
                    feature_buffer_size = 0

            except StopIteration:
                self._save_extracted_features(
                    self=self,
                    features=out_features,
                    targets=out_targets,
                    dist_rank=dist_rank,
                    chunk_index=chunk_index,
                    split=split_name,
                    output_folder=output_folder,
                    meta=out_meta,
                    dataset_name=data_source
                )
                break
