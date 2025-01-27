import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import json
import torch
import numpy as np
from pycocotools.coco import COCO
import torch.utils.data as data

from utils.transforms import normalize

class RGBXDataset(data.Dataset):
    def __init__(self, setting, split_name, preprocess=None, file_length=None):
        super(RGBXDataset, self).__init__()
        self._split_name = split_name
        self._rgb_path = setting['rgb_root']
        self._rgb_format = setting['rgb_format']
        self._x_path = setting['x_root']
        self._x_format = setting['x_format']
        self._x_single_channel = setting['x_single_channel']
        self.preprocess = preprocess
        self._file_length = file_length

        # Initialize COCO API
        self.coco = COCO(setting[f'{split_name}_json'])
        # print the amount of images in the dataset
        print(f"Amount of images in the dataset: {len(self.coco.getImgIds())}")
        self.imgIds = self.coco.getImgIds()
        self.catIds = [0] + self.coco.getCatIds()
        self.class_names = setting['class_names']

    def __len__(self):
        return self._file_length if self._file_length is not None else len(self.imgIds)

    def __getitem__(self, index):
        try:
            # Retrieve image metadata from COCO
            img_data = self.coco.loadImgs([self.imgIds[index-2]])[0]
                
            img_id = img_data['id']
            img_name = img_data['file_name']

            # Extract `video_id` and `frame_id` from `img_name`
            img_name_split = img_name.split('/')
            if len(img_name_split) == 3:
                video_id, video_type, frame_id = img_name_split
            elif len(img_name_split) == 2:
                video_id, frame_id = img_name_split
            
            # video_id, video_type = os.path.split(video_id)
            # remove the extension
            frame_id = frame_id.split('.')[0]

            # Paths for RGB and additional modality X
            rgb_path = os.path.join(self._rgb_path, video_id, 'color', f"{frame_id}{self._rgb_format}")
            x_path = os.path.join(self._x_path, video_id, 'depth', f"{frame_id}{self._x_format}")

            # Load images
            rgb = self._open_image(rgb_path, cv2.COLOR_BGR2RGB)
            x = self._open_image(x_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH if self._x_single_channel else cv2.COLOR_BGR2RGB)

            if rgb is None or x is None:
                raise FileNotFoundError(f"Image not found: {video_id}-{frame_id}")

            # if the rgb image is one channel, convert it to 3 channels
            if len(rgb.shape) == 2:
                rgb = cv2.merge([rgb, rgb, rgb])
            # if the x image is one channel, convert it to 3 channels
            if len(x.shape) == 2:
                x = cv2.merge([x, x, x])

            # Generate mask from COCO annotations
            ann_ids = self.coco.getAnnIds(imgIds=[img_id], catIds=self.catIds)
            anns = self.coco.loadAnns(ann_ids)
            gt = self._generate_mask(img_data['height'], img_data['width'], anns)

            # Apply preprocessing if any
            if self.preprocess is not None:
                rgb, gt, x = self.preprocess(rgb, gt, x)
            else:   # either way, normalize the images
                rgb = normalize(rgb, self.norm_mean, self.norm_std)
                x = normalize(x, self.norm_mean, self.norm_std)
            
            # Convert to tensors
            rgb = torch.from_numpy(np.ascontiguousarray(rgb)).float()*0.0
            gt = torch.from_numpy(np.ascontiguousarray(gt))
            x = torch.from_numpy(np.ascontiguousarray(x)).float()

            output_dict = dict(data=rgb, label=gt, modal_x=x, fn=img_name, n=len(self.imgIds))
            return output_dict

        except Exception as e:
            raise Exception(f'Error loading image {index}: {e}')


    def _generate_mask(self, height, width, anns):
        """Generate a segmentation mask from COCO annotations."""
        # create a blank mask
        mask = np.zeros((height, width), dtype=np.uint8)
        # per each annotation, draw that value on the mask
        for ann in anns:
            mask = np.maximum(self.coco.annToMask(ann) * ann['category_id'], mask)
        return mask

    def get_length(self):
        return len(self.imgIds)

    @staticmethod
    def _open_image(filepath, mode=cv2.IMREAD_COLOR):
        # always add cv2.IMREAD_UNCHANGED to read the image as it is
        if filepath.endswith('.exr'):
            image = cv2.imread(filepath, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            # map the image to 65535 by looking at the maximum value
            image = (image / image.max() * 65535).astype(np.uint16)
            return image
        else:
            return cv2.imread(filepath, mode | cv2.IMREAD_UNCHANGED)

    @staticmethod
    def _gt_transform(gt):
        return gt - 1 

    @classmethod
    def get_class_colors(*args):
        def uint82bin(n, count=8):
            """returns the binary of integer n, count refers to amount of bits"""
            return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])

        N = 41
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
        class_colors = cmap.tolist()
        return class_colors

