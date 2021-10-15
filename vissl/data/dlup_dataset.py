# coding=utf-8
# Copyright (c) HISSL Contributors

import logging
import pathlib
import time
from enum import Enum
from typing import List

import numpy as np
import PIL.Image
from hissl.utils.io import txt_of_paths_to_list


try:
    from dlup.background import AvailableMaskFunctions, get_mask
    from dlup.data.dataset import ConcatDataset, SlideImage, TiledROIsSlideImageDataset
    from dlup.tiling import TilingMode
except ImportError:
    raise ImportError("Make sure that DLUP is installed with 'vissl/third_party/dlup$ python setup.py develop'")

from vissl.config import AttrDict


class AvailableMaskTypes(Enum):
    func: str = "func"


class TransformDLUP2HISSL:
    """
    A small class to transform the objects returned by a DLUP dataset to the expected object by VISSL.
    Essentially, it ensures the image object is of type PIL.Image.Image, and ensures that the paths are strings.
    This is used in (the nki-ai fork of) vissl, in data.dlup_dataset.
    There, it only returns the image and an is_success flag.
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        # torch and VISSL collate functions can not handle a pathlib.path object, \
        # and want a string instead
        sample["path"] = str(sample["path"])
        # Openslide returns RGBA, but most neural networks want RGB
        sample["image"] = sample["image"].convert("RGB")
        return sample


class DLUPSlideImageDataset:
    """Class that wraps the DLUP SlideImageDataset class to be used by VISSL

    This class wraps the DLUP TiledROIsSlideImageDataset following the template as stated in
    https://vissl.readthedocs.io/en/v0.1.5/extend_modules/data_source.html

    It takes the necessary config variables and initalizes the SlideImageDataset which is stored as a variable.
    This class is called to get the length and to get items.

    Note that the VISSL transforms are still managed by VISSL with the ssl_transforms_wrapper, and that in the
    background this class is still wrapped by GenericSSLDataset

    Examples
    --------
    In the `config.yaml` file that you use for your experiment, set
    ```
    DATA:
        TRAIN:
            DATA_SOURCES: [dlup_wsi]
            DATASET_NAMES: [test_dlup_wsi_on_1_wsi]    or
            DATA_PATHS: /absolute/path/to/filenames.txt
        DLUP:
            ROOT_DIR: root dir from which paths in file passed in `path` can be found
            CROP: True
            MPP: 0.5
            TILE_SIZE:
                X: 512
                Y: 512
            TILE_OVERLAP:
                X: 0
                Y: 0
            TILE_MODE: skip  # skip, overflow, or fit
            MASK:
                USE_MASK: True
                MASK_TYPE: func
                MASK_FUNCTION:
                    NAME: improved_fesi
                FOREGROUND_THRESHOLD: 0.1
    ```
    """

    def __init__(self, cfg: AttrDict, data_source: str, path: str, split: str, dataset_name: str):
        """
        Parameters
        ----------
        cfg :
            The complete run configuration from the config.yaml file
        Required cfg parameters :
            DATA.DLUP.MASK.USE_MASK: bool = [True,False]
                Whether or not to do foreground segmentation

            If USE_MASK is True, the following cfg parameters are required:
                DATA.DLUP.MASK.MASK_TYPE: str = ["function", "file"]
                    Compute a mask using one of the implemented functions, or read a mask from file
                DATA.DLUP.MASK.MASK_FUNCTION.NAME: str = ["fesi", "improved_fesi"]
                    If DATA.DLUP.MASK.MASK_TYPE == "function", see which function to use from the avilable functions
                DATA.DLUP.MASK.MASK_FILE.PATH: str = "/absolute/path/to/file.txt"
                    Not yet implemented. Suggestion is to point to a .txt that holds paths to .ndarray objects
                    with a similar directory and file structure name as DATA.TRAIN.DATA_PATHS

                Note that MASK_FUNCTION.NAME and MASK_FILE.PATH are mutually exclusive

                DATA.DLUP.MASK.FOREGROUND_THRESHOLD: float = [0,1]
                    threshold as defined in dlup.background.is_foreground
            DATA.DLUP.MPP: float = mpp as defined in `SlideImageDataset`
            DATA.DLUP.TILE_SIZE.X: int = tile_size as defined in `SlideImageDataset`
            DATA.DLUP.TILE_SIZE.Y: int = tile_size as defined in `SlideImageDataset`
            DATA.DLUP.TILE_OVERLAP.X: int = overlap as defined in `SlideImageDataset`
            DATA.DLUP.TILE_OVERLAP.Y: int = overlap as defined in `SlideImageDataset`
            DATA.DLUP.TILE_MODE: str = tile mode as defined in `SlideImageDataset`
            DATA.DLUP.CROP: bool = crop as defined in `SlideImageDataset`
            optional, not implemented: DATA.DLUP.MASK: str = .txt file holding paths to masks for each WSI
            DATA.DLUP.FOREGROUND_THRESHOLD: float = threshold for the mask as defined in `SlideImageDataset`
            DATA.DLUP.ROOT_DIR: str = root dir from which paths in file passed in `path` can be found
        data_source :
            name of the type of datasource. should be dlup_wsi
        path :
            Path to a .txt file with relative paths to WSIs for this set and fold. Uses cfg.DATA.DLUP.ROOT_DIR
            as root dir to search for relative file paths from. This is passed through VISSL from DATA.TRAIN.DATA_PATHS
        split :
            TRAIN or TEST or VAL. We set it up so this is unused by HISSL, as the path should point to a .txt file
            holding all paths to the WSIs used for this split and fold
        dataset_name :
            the dataset name which can be registered through VISSL
        """
        assert data_source in [
            "disk_filelist",
            "disk_folder",
            "dlup_wsi",
        ], "data_source must be either disk_filelist or disk_folder or dlup_wsi"
        self.cfg = cfg
        self.split = split
        self.dataset_name = dataset_name
        self.data_source = data_source

        mpp = cfg["DATA"]["DLUP"].MPP
        tile_size = (cfg["DATA"]["DLUP"].TILE_SIZE.X, cfg["DATA"]["DLUP"].TILE_SIZE.Y)
        tile_overlap = (
            cfg["DATA"]["DLUP"].TILE_OVERLAP.X,
            cfg["DATA"]["DLUP"].TILE_OVERLAP.Y,
        )
        tile_mode = TilingMode[cfg["DATA"]["DLUP"].TILE_MODE]
        crop = cfg["DATA"]["DLUP"].CROP
        foreground_threshold = cfg["DATA"]["DLUP"]["MASK"].FOREGROUND_THRESHOLD
        root_dir = cfg["DATA"]["DLUP"].ROOT_DIR

        use_mask = cfg["DATA"]["DLUP"]["MASK"].USE_MASK

        if use_mask:
            logging.info("Using a mask on each WSI to remove background tiles.")
        else:
            logging.info("Using all tiles from the WSI, thus not removing background tiles.")

        if use_mask:
            mask_type = cfg["DATA"]["DLUP"]["MASK"].MASK_TYPE
            if mask_type not in AvailableMaskTypes.__members__.keys():
                logging.error(
                    f"{mask_type} is not yet implemented. Choose a type from {AvailableMaskTypes.__members__.keys()}"
                )
                raise ValueError

            logging.info(f"Using a {mask_type} for masking")

            if mask_type == "func":
                mask_function_name = cfg["DATA"]["DLUP"]["MASK"]["MASK_FUNCTION"].NAME
                if mask_function_name not in AvailableMaskFunctions.__members__.keys():
                    logging.error(
                        f"{mask_function_name} is not yet implemented. Choose a function from {AvailableMaskFunctions.__members__.keys()}"
                    )
                logging.info(f"Using {mask_function_name} to compute the tissue mask on the fly")
                mask_func = AvailableMaskFunctions[mask_function_name]

        path = pathlib.Path(path)
        relative_wsi_paths = txt_of_paths_to_list(path)

        # only transforms a Pathlib object to a string to work with the standard pytorch collate. VISSL has a
        # very involved collate functions, see vissl.data.collators, which we would rather not touch to manage
        # pathlib objects. Can easily add it though: https://vissl.readthedocs.io/en/v0.1.5/extend_modules/data_collators.html
        transform = TransformDLUP2HISSL()
        single_wsi_datasets: list = []
        for relative_wsi_path in relative_wsi_paths:
            absolute_wsi_path = root_dir / relative_wsi_path
            slide_image = SlideImage.from_file_path(absolute_wsi_path)
            mask = None
            if use_mask:
                if mask_type == "func":
                    t1 = time.time()
                    mask = get_mask(slide=slide_image, mask_func=mask_func)
                    t2 = time.time()
                    dt = int(t2 - t1)
                    logging.info(f"Computing mask for {relative_wsi_path} took {dt} seconds")
                else:
                    logging.error(
                        f"We currently only allow MASK_TYPE to be any of {AvailableMaskTypes.__members__.keys()}"
                    )
            single_wsi_datasets.append(
                TiledROIsSlideImageDataset.from_standard_tiling(
                    absolute_wsi_path,
                    mpp,
                    tile_size,
                    tile_overlap,
                    tile_mode,
                    crop,
                    mask,
                    foreground_threshold,
                    transform,
                )
            )
        self.dlup_dataset = ConcatDataset(single_wsi_datasets)

    def load_mask(self, mask: str) -> List[np.ndarray]:
        """
        Load the binary mask used to filter each region, as used in SlideImageDataset
        """
        # .txt -> List[pathlib.Path]
        # List[pathlib.Path] -> List[np.ndarray]
        # return List[np.ndarray]
        raise NotImplementedError

    def num_samples(self) -> int:
        """
        Size of the dataset
        """
        return len(self.dlup_dataset)  # Use the implementation from the DLUP class

    def __len__(self) -> int:
        """
        Size of the dataset
        """
        return self.num_samples()

    def __getitem__(self, index) -> (PIL.Image.Image, bool):
        # From https://vissl.readthedocs.io/en/latest/extend_modules/data_source.html:
        # is_success should be True or False indicating whether loading data was successful or failed
        # loaded data should be PIL.Image.Image if image data

        # We do not do a try-except that returns (None, False) for now. Since we do not know the
        # possible failure modes, we would like the system to throw errors for now.
        sample = self.dlup_dataset.__getitem__(index)
        image = sample["image"]
        is_success = True
        return image, is_success
