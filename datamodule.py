import os
import re
import csv
import glob
import json
import numpy as np

from pprint import pprint
from typing import Callable, Optional, Sequence, List
from argparse import ArgumentParser
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from lightning import seed_everything
seed = seed_everything(21, workers=True)

import torch
# from torch.utils.data import Dataset, DataLoader
from monai.config import IndexSelection, KeysCollection, SequenceStr
from monai.config import KeysCollection
from collections.abc import Mapping, Hashable
from monai.data.wsi_reader import WSIReader
from monai.data.wsi_datasets import PatchWSIDataset
from monai.data import CacheDataset, ThreadDataLoader
from monai.data import list_data_collate, pad_list_data_collate
from monai.utils import set_determinism
from monai.transforms import (
    apply_transform,
    Randomizable,
    Compose,
    OneOf,
    EnsureChannelFirstDict,
    LoadImageDict,
    CropDict,
    CenterSpatialCropDict,
    RandSpatialCropDict,
    RandAxisFlipDict, 
    RandRotate90Dict, 
    RandCropByPosNegLabelDict,
    RandSpatialCropSamplesDict,
    SpatialCropDict,
    SpacingDict,
    OrientationDict,
    DivisiblePadDict,
    CropForegroundDict,
    ResizeDict,
    Rotate90Dict,
    TransposeDict,
    RandFlipDict,
    RandZoomDict,
    ZoomDict,
    RandRotateDict,
    HistogramNormalizeDict,
    ScaleIntensityDict,
    ScaleIntensityRangeDict,
    # ToTensorDict,
    ToTensorDict,
)
from pytorch_lightning import LightningDataModule

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
def parse_filename(filename):
    """
    Parse the filename to extract features for classification.
    
    Args:
        filename (str): The filename to parse.
        
    Returns:
        list: A binary vector representing [Instrument, Magnification, Paraffin, Coverslip].
    """
    # Define the patterns to extract information
    instrument_pattern = r'(NanoZoomer|P250)'
    magnification_pattern = r'(\d{2})X'
    paraffin_pattern = r'(no_paraffin|paraffin)'
    # coverslip_pattern = r'(uncoverslip|coverslip)'
    coverslip_pattern = r'(aqueouscoverslip|nonaqueouscoverslip|uncoverslip)'

    # Initialize binary vector
    binary_vector = [0, 0, 0, 0]

    # Extract instrument
    instrument_match = re.search(instrument_pattern, filename)
    if instrument_match:
        instrument = instrument_match.group(1)
        # binary_vector[0] = 0 if instrument == 'NanoZoomer' else 1
        if instrument == 'NanoZoomer':
            binary_vector[0] = 0
        else:
            binary_vector[0] = 1

    # Extract magnification
    magnification_match = re.search(magnification_pattern, filename)
    if magnification_match:
        magnification = int(magnification_match.group(1))
        # binary_vector[1] = 0 if magnification == 20 else 1
        if magnification == 20:
            binary_vector[1] = 0
        else: # elif magnification == 40:
            binary_vector[1] = 1

    # Extract paraffin information
    paraffin_match = re.search(paraffin_pattern, filename)
    if paraffin_match:
        paraffin_type = paraffin_match.group(1)
        # binary_vector[2] = 0 if paraffin_type == 'no_paraffin' else 1
        if paraffin_type == 'no_paraffin':
            binary_vector[2] = 0
        else: # elif paraffin_type == 'paraffin':
            binary_vector[2] = 1

    # Extract coverslip information
    coverslip_match = re.search(coverslip_pattern, filename)
    if coverslip_match:
        coverslip_type = coverslip_match.group(1)
        # binary_vector[3] = 0 if coverslip_type == 'uncoverslip' else 1
        if coverslip_type == 'aqueouscoverslip':
            binary_vector[3] = 0
        elif coverslip_type == 'nonaqueouscoverslip':
            binary_vector[3] = 1
        elif coverslip_type == 'uncoverslip':
            binary_vector[3] = 2

    return binary_vector


class LongestAxisCropDict(CropDict):
    """
    Dictionary-based transform that crops an image tensor along the longest axis.

    Args:
        keys: keys of the corresponding items to be transformed.
        crop_choice: The crop choice (1, 2, or 3) indicating which part of the crop to take.
        allow_missing_keys: Don't raise an exception if a key is missing.
    """

    def __init__(self, keys: KeysCollection, crop_choice: int = 1, allow_missing_keys: bool = False, lazy: bool = False):
        super().__init__(keys=keys, cropper=None, allow_missing_keys=allow_missing_keys, lazy=lazy)
        if crop_choice not in [1, 2, 3]:
            raise ValueError("crop_choice must be 1, 2, or 3")
        self.crop_choice = crop_choice

    def __call__(self, data: Mapping[Hashable, torch.Tensor], lazy: bool = False) -> dict[Hashable, torch.Tensor]:
        d = dict(data)

        for key in self.key_iterator(d):
            image_tensor = d[key]

            # Ensure the image is a tensor
            if not isinstance(image_tensor, torch.Tensor):
                raise ValueError(f"Input must be a PyTorch tensor. Got {type(image_tensor)} instead.")

            # Debugging: Print the shape of the tensor
            # print(f"Shape of {key}: {image_tensor.shape}")

            # Check if the tensor has exactly 3 dimensions (C, H, W)
            if image_tensor.dim() != 3:
                raise ValueError(f"Input tensor must have exactly 3 dimensions (C, H, W). Got {image_tensor.dim()} dimensions.")

            # Get the shape of the image (C, H, W)
            c, h, w = image_tensor.shape

            # Determine the longest axis
            longest_axis = max(h, w)

            # Calculate crop size (one-third of the longest axis)
            if longest_axis == h:
                crop_size = (c, longest_axis // 3, w)  # Crop along height
                start_idx = (self.crop_choice - 1) * (longest_axis // 3)
                roi_start = (start_idx, 0)  # Start cropping from calculated index
                roi_end = (start_idx + (longest_axis // 3), w)  # End cropping at calculated index
                cropped_image = image_tensor[:, roi_start[0]:roi_end[0], roi_start[1]:roi_end[1]]
            else:
                crop_size = (c, h, longest_axis // 3)  # Crop along width
                start_idx = (self.crop_choice - 1) * (longest_axis // 3)
                roi_start = (0, start_idx)  # Start cropping from calculated index
                roi_end = (h, start_idx + (longest_axis // 3))  # End cropping at calculated index
                cropped_image = image_tensor[:, roi_start[0]:roi_end[0], roi_start[1]:roi_end[1]]

            # Replace the original image with the cropped one
            d[key] = cropped_image

        return d


class PairedDataset(CacheDataset, Randomizable):
    def __init__(
        self,
        keys: Sequence,
        data: Sequence,
        transform: Optional[Callable] = None,
        length: Optional[Callable] = None,
        batch_size: int = 32,
        is_training: bool = True,
    ) -> None:
        self.keys = keys
        self.data = data
        self.length = length
        self.transform = transform
        self.is_training = is_training

    def __len__(self) -> int:
        if self.length is None:
            return min((len(dataset) for dataset in self.data))
        else:
            return self.length

    def _transform(self, index: int):
        data = {}
        self.R.seed(index)
        
        if self.is_training:
            rand_idx = self.R.randint(0, len(self.data))
            data["imageA"] = self.data[rand_idx]["imageA"]
            data["imageB"] = self.data[rand_idx]["imageB"]
            data["labelA"] = self.data[rand_idx]["labelA"]
            data["labelB"] = self.data[rand_idx]["labelB"]
        else:
            fixed = self.R.randint(0, len(self.data))
            data["imageA"] = self.data[fixed]["imageA"]
            data["imageB"] = self.data[fixed]["imageB"]
            data["labelA"] = self.data[fixed]["labelA"]
            data["labelB"] = self.data[fixed]["labelB"]
            
        if self.transform is not None:
            data = apply_transform(self.transform, data)

        return data


class PairedDataModule(LightningDataModule):
    def __init__(
        self,
        root_folder: str,
        batch_size: int = 32,
        img_shape: int = 512,
        train_samples: int = 2000,
        val_samples: int = 400,
        test_samples: int | None = None,
    ):
        super().__init__()
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples
        self.img_shape = img_shape
        self.batch_size = batch_size
        
        # Find all subfolders in the root directory
        self.paired_folders = [f.path for f in os.scandir(root_folder) if f.is_dir()]
        
        # Initialize lists to hold valid image pairs
        self.image_pairs: List[dict] = []

        # Use glob to find all subdirectories containing "Training" and check for TIFF files
        if self.test_samples is None:
            folders = glob.glob(os.path.join(root_folder, '*/**/'), recursive=True)
        else:
            folders = [root_folder]
            # folders = glob.glob(os.path.join(root_folder, '*'), recursive=True)
        print(len(folders))
        for subdir in folders:
            image_files = glob.glob(os.path.join(subdir, "*.tiff"))
            # Check for "HE" in filenames and assign accordingly
            imageA, imageB = None, None
            for image_file in image_files:
                if "HE" not in os.path.basename(image_file):
                    imageA = image_file  # Assign to imageA if "HE" is not found
                    labelA = parse_filename(os.path.dirname(imageA))
                    labelA = np.array(labelA)
                else:
                    imageB = image_file  # Assign to imageB if "HE" is found
                    labelB = parse_filename(os.path.dirname(imageB))
                    labelB = np.array(labelB)
            
            
            # Ensure both images are assigned before appending
            if imageA and imageB:
                # Verify that both images have the same size
                with Image.open(imageA) as imgA, Image.open(imageB) as imgB:
                    if imgA.size == imgB.size:
                        self.image_pairs.append({
                            "imageA": imageA,
                            "imageB": imageB,
                            "labelA": labelB,
                            "labelB": labelB,
                        })
                    else:
                        print(f"Size mismatch: {imageA} size {imgA.size} vs {imageB} size {imgB.size}")
      
        print(self.image_pairs)
        print(f"Total pairs: {len(self.image_pairs)}")
        filename = 'data.json'
        with open(filename, 'w') as json_file:
            json.dump(self.image_pairs, json_file, indent=4, cls=NumpyEncoder)
            print(f"Saved image pairs to {filename}")

    def setup(self, seed: int = 2222, stage: Optional[str] = None):
        set_determinism(seed=seed)

    
    def train_dataloader(self):
        # Define transformations using MONAI
        transform_pipeline = Compose([
            LoadImageDict(keys=["imageA", "imageB"], image_only=True),
            EnsureChannelFirstDict(keys=["imageA", "imageB"]),
            # OneOf([
            #     LongestAxisCropDict(keys=["imageA", "imageB"], crop_choice=1),
            #     LongestAxisCropDict(keys=["imageA", "imageB"], crop_choice=3),
            # ]), 
            
            RandSpatialCropDict(keys=["imageA", "imageB"], roi_size=self.img_shape, random_size=False),
            RandAxisFlipDict(keys=["imageA", "imageB"], prob=0.75),
            RandRotate90Dict(keys=["imageA", "imageB"], prob=0.75),
            ScaleIntensityRangeDict(keys=["imageA", "imageB"], 
                clip=True,
                a_min=0.0,
                a_max=255.0,
                b_min=0.0,
                b_max=1.0,),
            # HistogramNormalizeDict(keys=["imageA", "imageB"], min=0.0, max=1.0,),
            ToTensorDict(keys=["imageA", "imageB", "labelB"]),
            # ToTensorDict(keys=["labelB"]),
        ])
      
        return ThreadDataLoader(
            PairedDataset(
                keys=["imageA", "imageB", "labelB"],
                data=self.image_pairs, 
                transform=transform_pipeline, 
                length=self.train_samples,
            ), 
            batch_size=self.batch_size,
            num_workers=16,
            collate_fn=pad_list_data_collate,
            shuffle=True,
        )
    
    def val_dataloader(self):
       # Define transformations using MONAI
        transform_pipeline = Compose([
            LoadImageDict(keys=["imageA", "imageB"], image_only=True),
            EnsureChannelFirstDict(keys=["imageA", "imageB"]),
            # OneOf([
            #     LongestAxisCropDict(keys=["imageA", "imageB"], crop_choice=2),
            # ]), 
            RandSpatialCropDict(keys=["imageA", "imageB"], roi_size=self.img_shape, random_size=False),
            ScaleIntensityRangeDict(keys=["imageA", "imageB"], 
                clip=True,
                a_min=0.0,
                a_max=255.0,
                b_min=0.0,
                b_max=1.0,),
            # HistogramNormalizeDict(keys=["imageA", "imageB"], min=0.0, max=1.0,),
            ToTensorDict(keys=["imageA", "imageB", "labelB"]),
            # ToTensorDict(keys=["labelB"]),
        ])
    
        return ThreadDataLoader(
            PairedDataset(
                keys=["imageA", "imageB", "labelB"],
                data=self.image_pairs, 
                transform=transform_pipeline, 
                length=self.val_samples,
            ), 
            batch_size=self.batch_size,
            num_workers=4,
            collate_fn=pad_list_data_collate,
            shuffle=False,
        )
    
    def test_dataloader(self):
       # Define transformations using MONAI
        transform_pipeline = Compose([
            LoadImageDict(keys=["imageA", "imageB"], image_only=True),
            EnsureChannelFirstDict(keys=["imageA", "imageB"]),
            OneOf([
                LongestAxisCropDict(keys=["imageA", "imageB"], crop_choice=2),
            ]), 
            RandSpatialCropDict(keys=["imageA", "imageB"], roi_size=self.img_shape, random_size=False),
            ScaleIntensityRangeDict(keys=["imageA", "imageB"], 
                clip=True,
                a_min=0.0,
                a_max=255.0,
                b_min=0.0,
                b_max=1.0,),
            # HistogramNormalizeDict(keys=["imageA", "imageB"], min=0.0, max=1.0,),
            ToTensorDict(keys=["imageA", "imageB", "labelB"]),
            # ToTensorDict(keys=["labelB"]),
        ])
    
        return ThreadDataLoader(
            PairedDataset(
                keys=["imageA", "imageB", "labelB"],
                data=self.image_pairs, 
                transform=transform_pipeline, 
                length=self.test_samples,
            ), 
            batch_size=self.batch_size,
            num_workers=16,
            collate_fn=pad_list_data_collate,
            shuffle=False,
        )

if __name__ == "__main__":
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument("--root_folder", type=str, required=True, help="Root folder containing subfolders with TIFF files.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")

    hparams = parser.parse_args()

    # Create paired data module
    datamodule = PairedDataModule(
        root_folder=hparams.root_folder,
        batch_size=hparams.batch_size,
        img_shape=256,
    )
    
    datamodule.setup()
    # for data in datamodule.val_dataloader():
    #     # print(data["imageA"].shape)
    #     # print(data["imageB"].shape)
    #     # print(data["labelB"].shape)
    #     pprint([
    #         data["imageA"],
    #         data["imageB"],
    #         data["labelA"],
    #         data["labelB"],
    #     ])
    #     print('\n')
    #     break

