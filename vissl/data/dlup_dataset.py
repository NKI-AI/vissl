# coding=utf-8
# Copyright (c) HISSL Contributors

import numpy as np

from typing import List

from hissl.utils.io import txt_of_paths_to_list

import pathlib

import PIL.Image

try:
    from dlup.data.dataset import TiledROIsSlideImageDataset, ConcatDataset
    from dlup.tiling import TilingMode
except:
    raise ImportError(
        "Make sure that DLUP is installed with 'vissl/third_party/dlup$ python setup.py develop'"
    )

from vissl.config import AttrDict


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
        sample["path"] = str(sample["path"])
        # torch and VISSL collate functions can not handle a pathlib.path object, \
        # and want a string instead
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
            DATASET_NAMES: [test_dlup_wsi_on_1_wsi]
        DLUP:
            CROP: True
            # MASK: None
            MASK_THRESHOLD: 0.1
            MPP: 0.5
            TILE_SIZE:
                X: 512
                Y: 512
            TILE_OVERLAP:
                X: 0
                Y: 0
            TILE_MODE: skip  # skip, overflow, or fit

    ```
    """

    def __init__(
        self, cfg: AttrDict, data_source: str, path: str, split: str, dataset_name: str
    ):
        """
        Parameters
        ----------
        cfg :
            The complete run configuration from the config.yaml file
        Required cfg parameters :
            DATA.DLUP.MPP: float = mpp as defined in `SlideImageDataset`
            DATA.DLUP.TILE_SIZE.X: int = tile_size as defined in `SlideImageDataset`
            DATA.DLUP.TILE_SIZE.Y: int = tile_size as defined in `SlideImageDataset`
            DATA.DLUP.TILE_OVERLAP.X: int = overlap as defined in `SlideImageDataset`
            DATA.DLUP.TILE_OVERLAP.Y: int = overlap as defined in `SlideImageDataset`
            DATA.DLUP.TILE_MODE: str = tile mode as defined in `SlideImageDataset`
            DATA.DLUP.CROP: bool = crop as defined in `SlideImageDataset`
            optional, not implemented: DATA.DLUP.MASK: str = .txt file holding paths to masks for each WSI
            DATA.DLUP.MASK_THRESHOLD: float = threshold for the mask as defined in `SlideImageDataset`
            DATA.DLUP.ROOT_DIR: str = root dir from which paths in file passed in `path` can be found
        data_source :
            name of the type of datasource. should be dlup_wsi
        path :
            Path to a .txt file with relative paths to WSIs for this set and fold. Uses cfg.DATA.DLUP.ROOT_DIR
            as root dir to search for relative file paths from
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
        mask_threshold = cfg["DATA"]["DLUP"].MASK_THRESHOLD
        root_dir = cfg["DATA"]["DLUP"].ROOT_DIR

        path = pathlib.Path(path)
        wsi_paths = txt_of_paths_to_list(path)

        try:
            mask = self.load_mask(
                cfg["DATA"]["DLUP"].MASK
            )  # raises NotImplementedError if called
        except AttributeError:
            mask = None  # if 'MASK' is not in config it is set to None, and we don't use a mask

        # only transforms a Pathlib object to a string to work with the standard pytorch collate. VISSL has a
        # very involved collate functions, see vissl.data.collators, which we would rather not touch to manage
        # pathlib objects
        transform = TransformDLUP2HISSL()

        single_wsi_datasets: list = [TiledROIsSlideImageDataset.from_standard_tiling(
            root_dir/wsi_path,
            mpp,
            tile_size,
            tile_overlap,
            tile_mode,
            crop,
            mask,
            mask_threshold,
            transform,
        ) for wsi_path in wsi_paths]

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
