import os
import os.path as osp
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, ConcatDataset
from PIL import Image, ImageFile
from .utils import BaseTransforms
import pdb

from nerv.utils import glob_all, load_obj

class ICSWMDataset(Dataset):
    """OBJ3D dataset from G-SWM."""

    def __init__(
        self,
        data_root,
        split,
        icswm_transform,
        n_sample_frames=6,
        frame_offset=None,
        video_len=50,
    ):

        assert split in ['train', 'val', 'test']

        #read file
        self.data_root = os.path.join(data_root, f'{split}.hdf5')
        self.data = h5py.File(self.data_root, 'r')
        self.transform = icswm_transform

        #the split
        self.split = split
        
        #sequence length
        self.n_sample_frames = n_sample_frames
        
        #the gap between two frames
        self.frame_offset = frame_offset
        
        #total video length to consider
        self.video_len = video_len

        # Get all numbers
        self.valid_idx = self._get_sample_idx()
        
        # by default, we load small video clips
        self.load_video = False


    def _read_bboxes(self, idx):
        """Load empty bbox and pres mask for compatibility."""
        bboxes = np.zeros((self.n_sample_frames, 5, 4))
        pres_mask = np.zeros((self.n_sample_frames, 5))
        return bboxes, pres_mask

    def _read_frames(self, idx):
        episode_id, frame_id = self.valid_idx[idx]
        episode_id, start_id = self.valid_idx[idx]
        frames = self.data['images'][episode_id, start_id:start_id + self.n_sample_frames]
        frames = (frames*255).astype(np.uint8)
        frames = [
            Image.fromarray(frames[n])
            for n in range(self.n_sample_frames)
        ]

        #the base transform
        for frame_id in range(len(frames)):
            frames[frame_id] = self.transform(frames[frame_id])
        frames = torch.stack(frames, dim=0) 
        return frames

    def get_video(self, video_idx):
        num_frames = (self.video_len + 1) // self.frame_offset
        frames = self.data['images'][video_idx,:num_frames,...]
        frames_t = (frames*255).astype(np.uint8)
        frames = [
            Image.fromarray(frames_t[n])
            for n in range(num_frames)
        ]
        frames = [self.transform(frame) for frame in frames]
        return {
            'video': torch.stack(frames, dim=0),
            'data_idx': video_idx,
        }

    def __getitem__(self, idx):
        """Data dict:
            - data_idx: int
            - img: [T, 3, H, W]
            - bbox: [T, max_num_obj, 4], empty, for compatibility
            - pres_mask: [T, max_num_obj], empty, for compatibility
        """
        if self.load_video:
            return self.get_video(idx)

        frames = self._read_frames(idx)
        data_dict = {
            'data_idx': idx,
            'img': frames,
        }
        if self.split != 'train':
            bboxes, pres_mask = self._read_bboxes(idx)
            data_dict['bbox'] = torch.from_numpy(bboxes).float()
            data_dict['pres_mask'] = torch.from_numpy(pres_mask).bool()
        return data_dict

    def _get_sample_idx(self):
        valid_idx = []  # (video_folder, start_idx)
        
        self.num_videos = self.data['images'].shape[0]
        self.files = [i for i in range(self.num_videos)]
        for episode_id in range(self.num_videos):
            if self.split == 'train':

                #vl-y = (sql-1)*nof + 1
                #y = vl - (sql-1)*nof - 1
                max_start_idx = self.video_len - \
                    (self.n_sample_frames - 1) * self.frame_offset
                
                #possible episodes
                valid_idx += [(episode_id, idx) for idx in range(max_start_idx)]
            
            # only test once per video
            else:
                valid_idx += [(episode_id, 0)]
        return valid_idx

    def __len__(self):
        if self.load_video:
            return len(self.num_videos)
        return len(self.valid_idx)


class ICSWMSlotsDataset(ICSWMDataset):
    """OBJ3D dataset from G-SWM with pre-computed slots."""

    def __init__(
        self,
        data_root,
        video_slots,
        split,
        icswm_transform,
        n_sample_frames=16,
        frame_offset=None,
        video_len=50,
    ):
        super().__init__(
            data_root=data_root,
            split=split,
            icswm_transform=icswm_transform,
            n_sample_frames=n_sample_frames,
            frame_offset=frame_offset,
            video_len=video_len,
        )

        # pre-computed slots
        self.video_slots = video_slots

    def _read_slots(self, idx):
        """Read video frames slots."""
        folder, start_idx = self.valid_idx[idx]
        slots = self.video_slots[folder]  # [T, N, C]
        slots = [
            slots[start_idx + n * self.frame_offset]
            for n in range(self.n_sample_frames)
        ]
        return np.stack(slots, axis=0).astype(np.float32)

    def __getitem__(self, idx):
        """Data dict:
            - data_idx: int
            - img: [T, 3, H, W]
            - bbox: [T, max_num_obj, 4], empty, for compatibility
            - pres_mask: [T, max_num_obj], empty, for compatibility
            - slots: [T, N, C] slots extracted from OBJ3D video frames
        """
        slots = self._read_slots(idx)
        frames = self._read_frames(idx)
        data_dict = {
            'data_idx': idx,
            'slots': slots,
            'img': frames,
        }
        if self.split != 'train':
            bboxes, pres_mask = self._read_bboxes(idx)
            data_dict['bbox'] = torch.from_numpy(bboxes).float()
            data_dict['pres_mask'] = torch.from_numpy(pres_mask).bool()
        return data_dict


def build_icswm_dataset(params, val_only=False):
    """Build OBJ3D video dataset."""

    args = dict(
        data_root=params.data_root,
        split='val',
        icswm_transform=BaseTransforms(params.resolution),
        n_sample_frames=params.n_sample_frames,
        frame_offset=params.frame_offset,
        video_len=params.video_len,
    )
    val_dataset = ICSWMDataset(**args)
    if val_only:
        return val_dataset
    args['split'] = 'train'
    train_dataset = ICSWMDataset(**args)

    return train_dataset, val_dataset


def build_icswm_slots_dataset(params, val_only=False):
    """Build OBJ3D video dataset with pre-computed slots."""
    slots = load_obj(params.slots_root)
    args = dict(
        data_root=params.data_root,
        video_slots=slots['val'],
        split='val',
        icswm_transform=BaseTransforms(params.resolution),
        n_sample_frames=params.n_sample_frames,
        frame_offset=params.frame_offset,
        video_len=params.video_len,
    )
    val_dataset = ICSWMSlotsDataset(**args)
    if val_only:
        return val_dataset
    args['split'] = 'train'
    args['video_slots'] = slots['train']
    train_dataset = ICSWMSlotsDataset(**args)
    return train_dataset, val_dataset
