import torch
import vissl
from vissl.data.data_helper import get_mean_image
from dlup.data.dataset import TiledSlideImageDataset, ConcatDataset
import pathlib

from vissl.data.dataset_catalog import VisslDatasetCatalog



class DlupTileDataset(ConcatDataset):
    """
    add documentation on how this dataset works

    Args:
        add docstrings for the parameters
    """

    def __init__(self, split, dataset_name):  #might need to keep the standard inputs cfg, data_source, path, split, dataset_name
        self.cfg = VisslDatasetCatalog.get(dataset_name)  #retrieves the dict with the paths from the catalog
        self.split = split
        self.dataset_name = dataset_name
        path_images = self.cfg[split][0]
        path_labels = self.cfg[split][1]
        self.path = pathlib.Path(path_images)

        super().__init__([TiledSlideImageDataset(_) for _ in self.path.glob("*")])

    def num_samples(self):
        """
        Size of the dataset
        """
        return self.__len__()

    def __getitem__(self, idx: int):
        """
        implement how to load the data corresponding to idx element in the dataset
        from your data source
        """
        tile_dict = ConcatDataset.__getitem__(self, idx)
        # is_success should be True or False indicating whether loading data was successful or failed
        # loaded data should be Image.Image if image data
        is_success = True

        return tile_dict["image"], is_success
