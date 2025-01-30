import os
import re
import csv
import glob
import json
import numpy as np
from copy import deepcopy
from typing import Any

from monai.config import IndexSelection, KeysCollection, SequenceStr
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.meta_tensor import MetaTensor
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.traits import LazyTrait, MultiSampleTrait
from monai.transforms.transform import LazyTransform, MapTransform, Randomizable

from pprint import pprint
from typing import Callable, Optional, Sequence, List, Dict
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
    RandSpatialCrop,
    RandCropDict,
    RandCropByPosNegLabel,
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


class RandCropByColorDict(RandSpatialCropSamplesDict):
    """
    Custom transform to randomly crop image areas based on RGB values in the label image,
    focusing on areas that contain a significant amount of any distinct color.
    """

    def __init__(
        self,
        keys: KeysCollection,
        roi_size: Sequence[int] | int,
        num_samples: int = 1,
        max_roi_size: Sequence[int] | int | None = None,
        random_center: bool = True,
        random_size: bool = False,
        allow_missing_keys: bool = False,
        lazy: bool = False,
        label_key: str | None = None,
        image_key: str | None = None,
        color_area_threshold: float = 0.1,  # Threshold for distinct color area
        color_diff_threshold: float = 0.1,  # Allowable difference between RGB channels
        
    ) -> None:
        super().__init__(keys, roi_size, num_samples, max_roi_size, random_center, random_size, allow_missing_keys=allow_missing_keys, lazy=lazy)
        self.label_key = label_key
        self.color_area_threshold = color_area_threshold
        self.color_diff_threshold = color_diff_threshold
        
    def check_color(self, label: torch.Tensor) -> bool:
        """
        If we look at an RGB color wheel, we see that purple and pink are next to each other. 
        On the other side of color wheel, we have yellow and green. 
        Since green is one of our 3 NumPy array RGB color channels, 
        filtering out pixels that have a high green channel value can be one way to potentially filter out parts of the slide that are not pink or purple. 
        This includes the white background, since white also has a high green channel value along with high red and blue channel values.
        https://github.com/CODAIT/deep-histopath/blob/master/docs/wsi-preprocessing-in-python/index.md
        """
        green_mask = label[[1]] > self.color_diff_threshold
        
        # Calculate total pixels in the patch
        total_area = label.shape[-1] * label.shape[-2]
        
        # Calculate area of distinct green pixels
        color_area = torch.logical_not(green_mask).sum().item()

        # Check if the area of distinct green pixels exceeds the specified threshold
        return (color_area / total_area) > self.color_area_threshold 
 
    def __call__(
        self, data: Mapping[Hashable, torch.Tensor], lazy: bool | None = None
    ) -> list[dict[Hashable, torch.Tensor]]:
        
        ret: list[dict[Hashable, torch.Tensor]] = [dict(data) for _ in range(self.cropper.num_samples)]
        # deep copy all the unmodified data
        for i in range(self.cropper.num_samples):
            for key in set(data.keys()).difference(set(self.keys)):
                ret[i][key] = deepcopy(data[key])

        # Flag to indicate if we should break the outer loop
        stop = False    
        for _ in range(10):
            # for each key we reset the random state to ensure crops are the same
            self.randomize()
            lazy_ = self.lazy if lazy is None else lazy
            for key in self.key_iterator(dict(data)):
                self.cropper.set_random_state(seed=self.sub_seed)
                for i, im in enumerate(self.cropper(data[key], lazy=lazy_)):   
                    ret[i][key] = im
                    if key==self.label_key and self.check_color(im):
                        stop = True
            if stop:
                break  # Break outer loop if condition met
        return ret
    
        # d = dict(data)
        # # the first key must exist to execute random operations
        # first_item = d[self.first_key(d)]
        # self.randomize(first_item.peek_pending_shape() if isinstance(first_item, MetaTensor) else first_item.shape[1:])
        # lazy_ = self.lazy if lazy is None else lazy
        # if lazy_ is True and not isinstance(self.cropper, LazyTrait):
        #     raise ValueError(
        #         "'self.cropper' must inherit LazyTrait if lazy is True "
        #         f"'self.cropper' is of type({type(self.cropper)}"
        #     )

        # max_attempts = 10  # Prevent infinite loop
        # attempts = 0
        # while attempts < max_attempts:
        #     for key in self.key_iterator(d):
        #         kwargs = {"randomize": False} if isinstance(self.cropper, Randomizable) else {}
        #         if isinstance(self.cropper, LazyTrait):
        #             kwargs["lazy"] = lazy_
        #         d[key] = self.cropper(d[key], **kwargs)  # type: ignore
        #         if key==self.label_key:
        #             label_patch = d[key]
        #             print(label_patch.shape)
        #     if self.check_color(label_patch):
        #         return d
        #     attempts += 1
        
        #     if attempts == max_attempts:
        #         print(f"Warning: Maximum attempts reached for sample {i}. Returning default data.")
        #         return d
     

    
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
            detr_idx = self.R.randint(0, len(self.data))
            data["imageA"] = self.data[detr_idx]["imageA"]
            data["imageB"] = self.data[detr_idx]["imageB"]
            data["labelA"] = self.data[detr_idx]["labelA"]
            data["labelB"] = self.data[detr_idx]["labelB"]
            
        if self.transform is not None:
            data = apply_transform(self.transform, data)

        return data


class PairedDataModule(LightningDataModule):
    def __init__(
        self,
        root_folder: str,
        batch_size: int = 32,
        img_shape: int = 512,
        train_folders: List[str] = None,
        val_folders: List[str] = None,
        test_folders: List[str] = None,
        train_samples: int = 2000,
        val_samples: int = 400,
        test_samples: Optional[int] = None,
    ):
        super().__init__()
        self.root_folder = root_folder
        self.batch_size = batch_size
        self.img_shape = img_shape
        self.train_folders = [os.path.join(self.root_folder, folder) for folder in train_folders] or []
        self.val_folders = [os.path.join(self.root_folder, folder) for folder in val_folders] or []
        self.test_folders = [os.path.join(self.root_folder, folder) for folder in test_folders] or []
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples
        
        # Initialize lists to hold valid image pairs for each dataset type
        self.train_image_pairs: List[Dict] = []
        self.valid_image_pairs: List[Dict] = []
        self.test_image_pairs: List[Dict] = []
        
        print(self.train_folders)
        print(self.val_folders)
        print(self.test_folders)
        # Load image pairs from specified folders
        self._load_image_pairs()

    def _load_image_pairs(self):
        # Load image pairs for training folders
        for subdir in self.train_folders:
            self._load_pairs_from_folder(subdir, self.train_image_pairs)

        # Load image pairs for validation folders
        for subdir in self.val_folders:
            self._load_pairs_from_folder(subdir, self.valid_image_pairs)

        # Load image pairs for testing folders (if specified)
        for subdir in self.test_folders:
            self._load_pairs_from_folder(subdir, self.test_image_pairs)

        print(f"Total training pairs found: {len(self.train_image_pairs)}")
        print(f"Total validation pairs found: {len(self.valid_image_pairs)}")
        print(f"Total testing pairs found: {len(self.test_image_pairs)}")

        # Save the paired images to a JSON file for reference (optional)
        all_image_pairs = {
            "train": self.train_image_pairs,
            "val": self.valid_image_pairs,
            "test": self.test_image_pairs
        }
        
        with open('data.json', 'w') as json_file:
            json.dump(all_image_pairs, json_file, indent=4, cls=NumpyEncoder)
            print("Saved image pairs to data.json")

    def _load_pairs_from_folder(self, folder: str, image_pair_list: List[Dict]):
        folders = glob.glob(os.path.join(folder, '*/**/'), recursive=True)
        for subdir in folders:
            image_files = glob.glob(os.path.join(subdir, "*.tiff"))
            
            imageA, imageB = None, None
            
            for image_file in image_files:
                if "HE" not in os.path.basename(image_file):
                    imageA = image_file  # Assign to imageA if "HE" is not found
                else:
                    imageB = image_file  # Assign to imageB if "HE" is found
            
            # Ensure both images are assigned before appending
            if imageA and imageB and self._images_have_same_size(imageA, imageB):
                labelA = parse_filename(os.path.dirname(imageA))
                labelB = parse_filename(os.path.dirname(imageB))
                image_pair_list.append({
                    "imageA": imageA,
                    "imageB": imageB,
                    "labelA": np.array(labelA),
                    "labelB": np.array(labelB),
                })

    def _images_have_same_size(self, imgA_path: str, imgB_path: str) -> bool:
        with Image.open(imgA_path) as imgA, Image.open(imgB_path) as imgB:
            if imgA.size != imgB.size:
                print(f"Size mismatch: {imgA_path} size {imgA.size} vs {imgB_path} size {imgB.size}")
                return False
            return True

    def setup(self, seed: int = 2222, stage: Optional[str] = None):
        set_determinism(seed=seed)

    def _get_dataloader(self, samples: Optional[int], shuffle: bool, data_type: str) -> ThreadDataLoader:
        randn_transform_pipeline = Compose([
            LoadImageDict(keys=["imageA", "imageB"], image_only=True),
            EnsureChannelFirstDict(keys=["imageA", "imageB"]),
            RandCropByColorDict(keys=["imageA", "imageB"], label_key="imageB", 
                color_area_threshold=0.2, color_diff_threshold=0.8,
                roi_size=self.img_shape, random_size=False),
            RandSpatialCropDict(keys=["imageA", "imageB"], roi_size=self.img_shape, random_size=False),
            RandAxisFlipDict(keys=["imageA", "imageB"], prob=0.75),
            RandRotate90Dict(keys=["imageA", "imageB"], prob=0.75),
            ScaleIntensityRangeDict(keys=["imageA", "imageB"], clip=True, a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
            ToTensorDict(keys=["imageA", "imageB", "labelB"]),
        ])

        fixed_transform_pipeline = Compose([
            LoadImageDict(keys=["imageA", "imageB"], image_only=True),
            EnsureChannelFirstDict(keys=["imageA", "imageB"]),
            RandSpatialCropDict(keys=["imageA", "imageB"], roi_size=self.img_shape, random_size=False),
            RandAxisFlipDict(keys=["imageA", "imageB"], prob=0.75),
            RandRotate90Dict(keys=["imageA", "imageB"], prob=0.75),
            ScaleIntensityRangeDict(keys=["imageA", "imageB"], clip=True, a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
            ToTensorDict(keys=["imageA", "imageB", "labelB"]),
        ])
        
        data_source = {
            'train': self.train_image_pairs,
            'val': self.valid_image_pairs,
            'test': self.test_image_pairs
        }[data_type]

        return ThreadDataLoader(
            PairedDataset(
                keys=["imageA", "imageB", "labelB"],
                data=data_source,
                transform=randn_transform_pipeline if data_type!="test" else fixed_transform_pipeline,
                length=samples,
            ),
            batch_size=self.batch_size,
            num_workers=8 if shuffle else 4,
            collate_fn=pad_list_data_collate,
            shuffle=shuffle,
        )

    def train_dataloader(self):
        return self._get_dataloader(samples=self.train_samples, shuffle=True, data_type='train')

    def val_dataloader(self):
        return self._get_dataloader(samples=self.val_samples, shuffle=True, data_type='val')

    def test_dataloader(self):
        return self._get_dataloader(samples=self.test_samples, shuffle=False, data_type='test')
    

def main():
    parser = ArgumentParser()
    parser.add_argument("--root_folder", 
        type=str, 
        required=True, 
        help="Root folder containing subfolders with TIFF files.", 
        default="data/ACT2_co-register")
    parser.add_argument("--batch_size", 
        type=int, 
        default=4, 
        help="Batch size.")
    
    # Add arguments for folder lists (optional)
    parser.add_argument("--train_folders", 
        type=str, 
        nargs='+', 
        help="List of training folder paths.",
        default="Subject2_Slide2,Subject4_Slide4AIN1")
    parser.add_argument("--val_folders", 
        type=str, 
        nargs='+', 
        help="List of validation folder paths.",
        default="Subject1_Slide1")
    parser.add_argument("--test_folders", 
        type=str, 
        nargs='+', 
        help="List of testing folder paths.", 
        default="Subject1_Slide1")

    hparams = parser.parse_args()

    # Create paired data module with specified arguments
    datamodule = PairedDataModule(
        root_folder=hparams.root_folder,
        batch_size=hparams.batch_size,
        train_folders=hparams.train_folders,
        val_folders=hparams.val_folders,
        test_folders=hparams.test_folders,
        img_shape=256  # Example shape; can be adjusted as needed.
    )

    datamodule.setup()

if __name__ == "__main__":
    main()

# if __name__ == "__main__":
#     from argparse import ArgumentParser
    
#     parser = ArgumentParser()
#     parser.add_argument("--root_folder", type=str, required=True, help="Root folder containing subfolders with TIFF files.")
#     parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")

#     hparams = parser.parse_args()

#     # Create paired data module
#     datamodule = PairedDataModule(
#         root_folder=hparams.root_folder,
#         batch_size=hparams.batch_size,
#         img_shape=256,
#     )
    
#     datamodule.setup()
#     # for data in datamodule.val_dataloader():
#     #     # print(data["imageA"].shape)
#     #     # print(data["imageB"].shape)
#     #     # print(data["labelB"].shape)
#     #     pprint([
#     #         data["imageA"],
#     #         data["imageB"],
#     #         data["labelA"],
#     #         data["labelB"],
#     #     ])
#     #     print('\n')
#     #     break

