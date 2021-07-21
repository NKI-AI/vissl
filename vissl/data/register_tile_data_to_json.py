
data_folder= "tcga_tile_folder"
train_imgs_path= "/mnt/archive/projectdata/data_tcga/duct_detection_tiling_20210719_512_2mpp"
json_data = {
        data_folder: {
            "train": [train_imgs_path, "<lbl_path>"],
        }
    }
from vissl.utils.io import save_file
save_file(json_data, "../../configs/config/dataset_catalog.json", append_to_json=False)
from vissl.data.dataset_catalog import VisslDatasetCatalog
print(VisslDatasetCatalog.list())
print(VisslDatasetCatalog.get(data_folder))


#val_imgs_path = "/mnt/archive/projectdata/data_tcga/test/test"
#            "val": [val_imgs_path, "<lbl_path>"]
