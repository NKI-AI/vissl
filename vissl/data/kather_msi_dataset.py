# coding=utf-8
# Copyright (c) HISSL Contributors

from vissl.data.disk_dataset import DiskImageDataset
from pathlib import Path


class KatherMSIDataset(DiskImageDataset):
    """
    Dataset class that is specific to the organization of https://zenodo.org/record/2530835
    It expects that files are structured as
    /path/to/root/TRAIN/MSS/*.png
    /path/to/root/TRAIN/MSS/*.png
    /path/to/root/TEST/MSIMUT/*.png
    /path/to/root/TEST/MSIMT/*.png
    """

    def __init__(self, *args, **kwargs):
        # Essentially, we're using a disk filelst, except that our getitem returns some metadata
        kwargs['data_source'] = 'disk_filelist'
        super(KatherMSIDataset, self).__init__(*args, **kwargs)

    def __getitem__(self, idx: int):
        """
        implement how to load the data corresponding to idx element in the dataset
        from your data source
        """
        loaded_data, is_success = super().__getitem__(idx)

        image_path = self.image_dataset[idx]

        # blk-YYYQNFFCQMAT-TCGA-AY-A71X-01Z-00-DX1.png
        # ^---------------^                                 unique identifier for the tile
        #                  ^-----------^                    case identifier
        #                  ^----------------------^         slide identifier

        # Extract patient ID from tile name
        filename_without_extension = Path(image_path).stem
        slide_id = str.join('-', filename_without_extension.split('-')[2:])
        case_id = str.join('-', filename_without_extension.split('-')[2:5])

        meta = {'path': image_path,
                'slide_id': slide_id,
                'case_id': case_id}

        # is_success should be True or False indicating whether loading data was successful or failed
        # loaded data should be Image.Image if image data
        return loaded_data, is_success, meta
