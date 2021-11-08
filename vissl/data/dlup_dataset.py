# coding=utf-8
# Copyright (c) HISSL Contributors

import logging
import pathlib
import time

import PIL.Image
from hissl.utils.io import txt_of_paths_to_list

try:
    from dlup.background import AvailableMaskFunctions, get_mask, load_mask
    from dlup.data.dataset import ConcatDataset, SlideImage, TiledROIsSlideImageDataset
    from dlup.tiling import TilingMode
    from dlup import DlupUnsupportedSlideError
except ImportError:
    raise ImportError("Make sure that DLUP is installed with 'vissl/third_party/dlup$ python setup.py develop'")

from vissl.config import AttrDict


class MaskGetter:
    """
    Takes the VISSL config and sets all parameters required for getting a mask for the dataset
    """
    def __init__(self, args):
        self.mask_args = args["DATA"]["DLUP"].MASK_PARAMS
        self.mask_options = {
            'compute_fesi': self.compute_fesi,
            'compute_improved_fesi': self.compute_improved_fesi,
            'load_from_disk': self.load_from_disk,
            'no_mask': self.no_mask
        }

        if 'MASK_FACTORY' not in self.mask_args.keys():
            logging.info("No mask factory set. Not using any tissue mask.")
            self.mask_args.MASK_FACTORY = 'no_mask'
        else:
            if self.mask_args.MASK_FACTORY not in self.mask_options.keys():
                logging.error(f"{self.mask_args.MASK_FACTORY} is not an available mask factory. Please choose any of {self.mask_options.keys()}")
                raise ValueError
        #
        if self.mask_args.MASK_FACTORY == 'load_from_disk':
            self.master_mask_file_path = pathlib.Path(self.mask_args.MASK_FILE_PATHS)
            self.mask_file_paths = txt_of_paths_to_list(self.master_mask_file_path)
            assert pathlib.Path(self.mask_args.MASK_ROOT).is_dir()
        else:
            self.mask_file_paths = None

        self.current_slide_image = None
        self.current_idx = None

        logging.info(f"Using {self.mask_args.MASK_FACTORY} on each WSI to remove background tiles.")

    def return_mask_from_config(self, slide_image, idx):
        """
        Returns a mask with the given mask_factory
        """
        self.current_idx = idx
        mask = self.mask_options[self.mask_args.MASK_FACTORY](slide_image=slide_image)
        return mask

    def compute_fesi(self, slide_image):
        """
        Compute mask on the fly with fesi
        """
        mask = self.compute_mask('fesi', slide_image)
        return mask

    def compute_improved_fesi(self, slide_image):
        """
        Compute mask on the fly with improved_fesi
        """
        mask = self.compute_mask('improved_fesi', slide_image)
        return mask

    def compute_mask(self, mask_function_name, slide_image):
        """
        Computes mask with any of the functions available
        """
        t1 = time.time()
        mask_func = AvailableMaskFunctions[mask_function_name]
        mask = get_mask(slide=slide_image, mask_func=mask_func)
        t2 = time.time()
        dt = int(t2 - t1)
        logging.info(f"Computing mask took {dt} seconds")
        return mask

    def load_from_disk(self, *args, **kwargs):
        """
        Loads mask from disk. Reads a .png saved by DLUP and converts into a npy object
        """
        mask = load_mask(mask_file_path=pathlib.Path(self.mask_args.MASK_ROOT) / self.mask_file_paths[self.current_idx])
        return mask

    def no_mask(self, *args, **kwargs):
        """
        Returns no mask
        """
        return None


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
            MASK_PARAMS:
                # Everything is currently only implemented to work on a single TRAIN set.
                MASK_FACTORY: "no_mask", "compute_fesi", "compute_improved_fesi", "load_from_disk"
                FOREGROUND_THRESHOLD: 0.1
                # Below only required for "load_from_disk", which we recommend as computing masks can take 1-3mins/WSI
                MASK_ROOT: /absolute/path/to/dir/the/dir/from/which/paths/of/MASK_FILE_PATHS.txt/are/created
                # We assume that the order of the masks matches the order of the WSIs
                MASK_FILE_PATHS: /absolute/path/to/txt/with/relative/paths.txt

    ```
    """

    def __init__(self, cfg: AttrDict, data_source: str, path: str, split: str, dataset_name: str):
        """
        Parameters
        ----------
        cfg :
            The complete run configuration from the config.yaml file
        Required cfg parameters :
            DATA.DLUP.MASK_PARAMS.MASK_FACTORY: str = ["no_mask", "compute_fesi", "compute_improved_fesi", "load_from_disk"]
            # If unset, "no_mask" will be used
            # If "load_from_disk" is used, requires
            DATA.DLUP.MASK_PARAMS.MASK_ROOT: str = /absolute/path/to/dir/the/dir/from/which/paths/of/MASK_FILE_PATHS.txt/are/created
            DATA.DLUP.MASK_PARAMS.MASK_FILE_PATHS: str = /absolute/path/to/txt/with/relative/paths.txt
            DATA.DLUP.MASK_PARAMS.FOREGROUND_THRESHOLD: float = [0,1]
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

        # --------------------------
        # Set main variables used
        # --------------------------
        self.cfg = cfg
        self.split = split
        self.dataset_name = dataset_name
        self.data_source = data_source
        self.mpp = self.cfg["DATA"]["DLUP"].MPP
        self.tile_size = (
            self.cfg["DATA"]["DLUP"].TILE_SIZE.X,
            self.cfg["DATA"]["DLUP"].TILE_SIZE.Y
        )
        self.tile_overlap = (
            self.cfg["DATA"]["DLUP"].TILE_OVERLAP.X,
            self.cfg["DATA"]["DLUP"].TILE_OVERLAP.Y,
        )
        self.tile_mode = TilingMode[cfg["DATA"]["DLUP"].TILE_MODE]
        self.crop = cfg["DATA"]["DLUP"].CROP

        self.root_dir = self.cfg["DATA"]["DLUP"].ROOT_DIR
        path = pathlib.Path(path)
        self.relative_wsi_paths = txt_of_paths_to_list(path)

        # --------------------------
        # Set transform
        # --------------------------
        # only transforms a Pathlib object to a string to work with the standard pytorch collate. VISSL has a
        # very involved collate functions, see vissl.data.collators, which we would rather not touch to manage
        # pathlib objects. Can easily add it though: https://vissl.readthedocs.io/en/v0.1.5/extend_modules/data_collators.html
        self.transform = TransformDLUP2HISSL()

        # --------------------------
        # Init the class that takes care of the mask options
        # --------------------------
        if self.cfg["DATA"]["DLUP"]["MASK_PARAMS"].MASK_FACTORY != "no_mask":
            self.foreground_threshold = self.cfg["DATA"]["DLUP"]["MASK_PARAMS"].FOREGROUND_THRESHOLD
        else:
            self.foreground_threshold = 0.1  # DLUP dataset erroneously requires a float instead of optional None
        self.mask_getter = MaskGetter(args=self.cfg)

        # --------------------------
        # Build dataset
        # --------------------------
        single_wsi_datasets: list = []
        logging.info(f"Building dataset...")
        for idx, relative_wsi_path in enumerate(self.relative_wsi_paths):
            absolute_wsi_path = self.root_dir / relative_wsi_path
            try:
                slide_image = SlideImage.from_file_path(absolute_wsi_path)
            except DlupUnsupportedSlideError:
                logging.warning(f"{absolute_wsi_path} is unsupported. Skipping WSI.")
                continue
            mask = self.mask_getter.return_mask_from_config(slide_image, idx)
            single_wsi_datasets.append(
                TiledROIsSlideImageDataset.from_standard_tiling(
                    path=absolute_wsi_path,
                    mpp=self.mpp,
                    tile_size=self.tile_size,
                    tile_overlap=self.tile_overlap,
                    tile_mode=self.tile_mode,
                    crop=self.crop,
                    mask=mask,
                    mask_threshold=self.foreground_threshold,
                    transform=self.transform,
                )
            )
        self.dlup_dataset = ConcatDataset(single_wsi_datasets)
        logging.info(f"Built dataset successfully")

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
        meta = {
            'path': sample["path"],
            'x': sample["coordinates"][0],
            'y': sample["coordinates"][1],
            'mpp': sample["mpp"],
            'w': sample["region_size"][0],
            'h': sample["region_size"][1],
            'region_index': sample['region_index']
        }
        is_success = True
        return image, is_success, meta
