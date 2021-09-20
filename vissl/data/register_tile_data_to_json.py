data_folder= "tcga_tile_folder"
data_folder_lisa= "tcga_tile_folder_lisa"
train_imgs_path= "/mnt/archive/projectdata/data_tcga/duct_detection_tiling_20210719_512_2mpp"
train_imgs_path_lisa= "/home/sdoyle/duct_detection_tiling_20210719_512_2mpp"

json_data = {
        data_folder: {
            "train": [train_imgs_path, "<lbl_path>"],
        },
    data_folder_lisa: {
        "train": [train_imgs_path_lisa, "<lbl_path>"],
    }
    }
from vissl.utils.io import save_file
save_file(json_data, "../../configs/config/dataset_catalog.json", append_to_json=False)
from vissl.data.dataset_catalog import VisslDatasetCatalog

VisslDatasetCatalog.register_data(name=data_folder_lisa, data_dict=json_data[data_folder_lisa])

print(VisslDatasetCatalog.list())
print(VisslDatasetCatalog.get(data_folder_lisa))


#val_imgs_path = "/mnt/archive/projectdata/data_tcga/test/test"
