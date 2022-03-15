# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from vissl.data.ssl_dataset import GenericSSLDataset, _convert_lbl_to_long


class GenericSSLDatasetHissl(GenericSSLDataset):
    """
    Subclass that also returns metadata of the sample returned by DLUPSlideImageDataset
    """

    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx: int):
        """
        Get the input sample for the minibatch for a specified data index.
        For each data object (if we are loading several datasets in a minibatch),
        we get the sample: consisting of {
            - image data,
            - label (if applicable) otherwise idx
            - data_valid: 0 or 1 indicating if the data is valid image
            - data_idx : index of the data in the dataset for book-keeping and debugging
            - meta : {path, mpp, x, y, w, h} if the data_source==dlup_wsi, {None} otherwise
        }

        Once the sample data is available, we apply the data transform on the sample.

        The final transformed sample is returned to be added into the minibatch.
        """

        if not self._labels_init and len(self.label_sources) > 0:
            self._load_labels()
            self._labels_init = True

        subset_idx = idx
        if self.data_limit >= 0 and self._can_random_subset_data_sources():
            if not self._subset_initialized:
                self._init_image_and_label_subset()
            subset_idx = self.image_and_label_subset[idx]

        # TODO: this doesn't yet handle the case where the length of datasets
        # could be different.
        item = {"data": [], "data_valid": [], "data_idx": [], "meta": []}
        for data_source in self.data_objs:
            if type(data_source).__name__ in ["DLUPSlideImageDataset", "KatherMSIDataset"]:
                data, valid, meta = data_source[subset_idx]
                item["meta"].append(meta)
            else:
                data, valid = data_source[subset_idx]
            item["data"].append(data)
            item["data_idx"].append(idx)
            item["data_valid"].append(1 if valid else -1)


        # There are three types of label_type (data labels): "standard",
        # "sample_index", and "zero". "standard" uses the labels associated
        # with a data set (e.g. directory names). "sample_index" assigns each
        # sample a label that corresponds to that sample's index in the
        # dataset (first sample will have label == 0, etc.), and is used for
        # SSL tasks in which the label is arbitrary. "zero" assigns
        # each sample the label == 0, which is necessary when using the
        # CutMixUp collator because of the label smoothing that is built in
        # to its functionality.
        if (len(self.label_objs) > 0) or self.label_type == "standard":
            item["label"] = []
            for label_source in self.label_objs:
                if isinstance(label_source, list):
                    lbl = [entry[subset_idx] for entry in label_source]
                else:
                    lbl = _convert_lbl_to_long(label_source[subset_idx])
                item["label"].append(lbl)
        elif self.label_type == "sample_index":
            item["label"] = []
            for _ in range(len(self.data_objs)):
                item["label"].append(idx)
        elif self.label_type == "zero":
            item["label"] = []
            for _ in range(len(self.data_objs)):
                item["label"].append(0)
        else:
            raise ValueError(f"Unknown label type: {self.label_type}")

        # apply the transforms on the image
        if self.transform:
            item = self.transform(item)
        return item
