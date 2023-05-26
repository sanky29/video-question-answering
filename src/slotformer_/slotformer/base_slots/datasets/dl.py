import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

from nerv.utils import load_obj, strip_suffix, read_img, VideoReader

from .utils import compact, BaseTransforms, BaseTransformsOur, anno2mask, masks_to_boxes_pad
import pdb

class CLEVRERDataset(Dataset):
    """Dataset for loading CLEVRER videos."""

    def __init__(
        self,
        data_root,
        clevrer_transforms,
        split='train',
        max_n_objects=6,
        video_len=128,
        n_sample_frames=6,
        warmup_len=5,
        frame_offset=None,
        load_mask=False,
        filter_enter=False,
    ):
        super().__init__()
        self.data_root = data_root
        self.split = split
        assert self.split in ['train', 'val', 'test']
        if(self.split == 'train' or self.split == 'val'):
            self.video_path = os.path.join(data_root, 'train', 'frames')
        else:
            self.video_path = os.path.join(data_root, 'test', 'frames')
        
        self.clevrer_transforms = clevrer_transforms
        self.max_n_objects = max_n_objects
        self.video_len = video_len
        self.n_sample_frames = n_sample_frames
        self.warmup_len = warmup_len
        self.frame_offset = video_len // n_sample_frames if \
            frame_offset is None else frame_offset
        
        self.num_videos = 0
        self.files = os.listdir(self.video_path)
        self.files.sort()
        
        #read files
        self.batches = []
        for f in self.files:
            f = os.path.join(self.video_path, f)
            data_file = h5py.File(f, "r")
            self.batches.append(data_file)
            self.num_videos += len(list(data_file.keys()))
        
        self.valid_idx = self._get_sample_idx()

        # by default, we load small video clips
        self.load_video = False


    def _rand_another(self, is_video=False):
        """Random get another sample when encountering loading error."""
        if is_video:
            another_idx = np.random.choice(self.num_videos)
            return self.get_video(another_idx)
        another_idx = np.random.choice(len(self))
        return self.__getitem__(another_idx)

    def _get_video_start_idx(self, idx):
        return self.valid_idx[idx]

    def _read_frames(self, idx):
        """Read video frames. Directly read from jpg images if possible."""
        video_idx, start_idx = self._get_video_start_idx(idx)
        videoidx = self.files[video_idx]
        batchidx = videoidx // 1000
        frames = torch.tensor(np.array(self.batches[batchidx][str(videoidx)][start_idx:n*self.frame_offset:self.frame_offset,...]))
        return self.clevrer_transforms(frames).permute(0,2,3,1)
        
        
    def __getitem__(self, idx):
        """Data dict:
            - data_idx: int
            - img: [T, 3, H, W]
            - error_flag: whether loading `idx` causes error and _rand_another
            - mask: [T, H, W], 0 is background
            - pres_mask: [T, max_num_obj], valid index of objects
            - bbox: [T, max_num_obj, 4], object bbox padded to `max_num_obj`
        """
        if self.load_video:
            return self.get_video(idx)

        frames = self._read_frames(idx)
        data_dict = {
            'data_idx': idx,
            'img': frames,
            'error_flag': False,
        }
        return data_dict

    def __len__(self):
        if self.load_video:
            return len(self.files)
        return len(self.valid_idx)

    def get_video(self, video_idx):
        videoidx = self.files[video_idx]
        batchidx = videoidx // 1000
        frames = torch.tensor(np.array(self.batches[batchidx][str(videoidx)]))
        frames = self.clevrer_transforms(frames).permute(0,2,3,1)
        return {
            'video': frames,
            'error_flag': False,
            'data_idx': video_idx,
        }

class CLEVRERSlotsDataset(CLEVRERDataset):
    """Dataset for loading CLEVRER videos and pre-computed slots."""

    def __init__(
        self,
        data_root,
        video_slots,
        clevrer_transforms,
        split='train',
        max_n_objects=6,
        video_len=128,
        n_sample_frames=10 + 6,
        warmup_len=5,
        frame_offset=None,
        load_img=False,
        load_mask=False,
        filter_enter=True,
    ):
        self.load_img = load_img
        self.load_mask = load_mask

        super().__init__(
            data_root=data_root,
            clevrer_transforms=clevrer_transforms,
            split=split,
            max_n_objects=max_n_objects,
            video_len=video_len,
            n_sample_frames=n_sample_frames,
            warmup_len=warmup_len,
            frame_offset=frame_offset,
            load_mask=load_mask,
            filter_enter=filter_enter,
        )

        # pre-computed slots
        self.video_slots = video_slots

    def _rand_another(self, is_video=False):
        """Random get another sample when encountering loading error."""
        if is_video:
            another_idx = np.random.choice(self.num_videos)
            return self.get_video(another_idx)
        another_idx = np.random.choice(len(self))
        return self.__getitem__(another_idx)

    def _read_slots(self, idx):
        """Read video frames slots."""
        video_idx, start_idx = self._get_video_start_idx(idx)
        video_path = self.files[video_idx]
        try:
            slots = self.video_slots[os.path.basename(video_path)]  # [T, N, C]
        except KeyError:
            raise ValueError
        slots = [
            slots[start_idx + n * self.frame_offset]
            for n in range(self.n_sample_frames)
        ]
        return np.stack(slots, axis=0).astype(np.float32)

    def __getitem__(self, idx):
        """Data dict:
            - data_idx: int
            - img: [T, 3, H, W]
            - slots: [T, N, C] slots extracted from CLEVRER video frames
            - error_flag: whether loading `idx` causes error and _rand_another
            - mask: [T, H, W], 0 is background
            - pres_mask: [T, max_num_obj], valid index of objects
            - bbox: [T, max_num_obj, 4], object bbox padded to `max_num_obj`
        """
        try:
            slots = self._read_slots(idx)
            data_dict = {
                'data_idx': idx,
                'slots': slots,
                'error_flag': False,
            }
            if self.load_img:
                data_dict['img'] = self._read_frames(idx)
            if self.load_mask:
                data_dict['mask'], data_dict['pres_mask'], \
                    data_dict['bbox'] = self._read_masks(idx)
        # empty video
        except ValueError:
            data_dict = self._rand_another()
            data_dict['error_flag'] = True
        return data_dict


def build_dl_dataset(params, val_only=False, test_set=False):
    """Build CLEVRER video dataset."""
    args = dict(
        data_root=params.data_root,
        clevrer_transforms=BaseTransforms(params.resolution),
        split='val',
        max_n_objects=6,
        n_sample_frames=params.n_sample_frames,
        warmup_len=params.input_frames,
        frame_offset=params.frame_offset,
        load_mask=params.get('load_mask', False),
        filter_enter=params.filter_enter,
    )

    if test_set:
        assert not val_only
        args['split'] = 'test'
        return CLEVRERDataset(**args)

    val_dataset = CLEVRERDataset(**args)
    if val_only:
        return val_dataset
    args['split'] = 'train'
    train_dataset = CLEVRERDataset(**args)
    return train_dataset, val_dataset


def build_dl_slots_dataset(params, val_only=False):
    """Build CLEVRER video dataset with pre-computed slots."""
    slots = load_obj(params.slots_root)
    args = dict(
        data_root=params.data_root,
        video_slots=slots['val'],
        clevrer_transforms=BaseTransformsOur(params.resolution),
        split='val',
        max_n_objects=6,
        n_sample_frames=params.n_sample_frames,
        warmup_len=params.input_frames,
        frame_offset=params.frame_offset,
        load_img=params.load_img,
        load_mask=params.get('load_mask', False),
        filter_enter=params.filter_enter,
    )
    val_dataset = CLEVRERSlotsDataset(**args)
    if val_only:
        return val_dataset
    args['split'] = 'train'
    args['video_slots'] = slots['train']
    train_dataset = CLEVRERSlotsDataset(**args)
    return train_dataset, val_dataset
