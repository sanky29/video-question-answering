import os
import sys
import pdb
import argparse
import importlib
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data._utils.collate import default_collate
import torchvision.utils as vutils
import wandb

os.environ["WANDB_API_KEY"] = 'c6b090662f8d8a4566bda49228d4a49b3d450a6f'
os.environ["WANDB_MODE"] = "offline"

from nerv.utils import dump_obj, mkdir_or_exist

from models import build_model
from datasets import build_dataset, build_clevrer_dataset
from models import to_rgb_from_tensor


'''
we want this for extracting visuals from the 
slots
'''
class SAViMethod:
    """SAVi model training method."""

    def __init__(self):
        pass

    def _make_video_grid(self, imgs, recon_combined, recons, masks):
        """Make a video of grid images showing slot decomposition."""
        # pause the video on the 1st frame in PHYRE

        # in PHYRE if the background is black, we scale the mask differently
        scale = 0. 
        # combine images in a way so we can display all outputs in one grid
        # output rescaled to be between 0 and 1
        out = to_rgb_from_tensor(
            torch.cat(
                [
                    imgs.unsqueeze(1),  # original images
                    recon_combined.unsqueeze(1),  # reconstructions
                    recons * masks + (1. - masks) * scale,  # each slot
                ],
                dim=1,
            ))  # [T, num_slots+2, 3, H, W]
        # stack the slot decomposition in all frames to a video
        save_video = torch.stack([
            vutils.make_grid(
                out[i].cpu(),
                nrow=out.shape[1],
                pad_value=1. - scale,
            ) for i in range(recons.shape[0])
        ])  # [T, 3, H, (num_slots+2)*W]
        return save_video

    @torch.no_grad()
    def _sample_video(self, model, dataset):
        """model is a simple nn.Module, not warpped in e.g. DataParallel."""
        model.eval()
        dst = dataset
        sampled_idx = self._get_sample_idx(5, dst)
        results, labels = [], []
        for i in sampled_idx:
            data_dict = dst.get_video(i.item())
            video, label = data_dict['video'].float().cuda(), \
                data_dict.get('label', None)  # label for PHYRE
            in_dict = {'img': video[None]}
            out_dict = model(in_dict)
            out_dict = {k: v[0] for k, v in out_dict.items()}
            recon_combined, recons, masks = out_dict['post_recon_combined'], \
                out_dict['post_recons'], out_dict['post_masks']
            imgs = video.type_as(recon_combined)
            save_video = self._make_video_grid(imgs, recon_combined, recons,
                                               masks)
            results.append(save_video)
            labels.append(label)

        if all(lbl is not None for lbl in labels):
            caption = '\n'.join(
                ['Success' if lbl == 1 else 'Fail' for lbl in labels])
        else:
            caption = None
        wandb.log({'val/video': self._convert_video(results, caption=caption)},
                  step=100)
        torch.cuda.empty_cache()

    def _get_sample_idx(self, N, dst):
        """Load videos uniformly from the dataset."""
        dst_len = len(dst.files)  # treat each video as a sample
        N = N - 1 if dst_len % N != 0 else N
        sampled_idx = torch.arange(0, dst_len, dst_len // N)
        return sampled_idx
    
    def _convert_video(self, video, caption=None):
        video = torch.cat(video, dim=2)  # [T, 3, B*H, L*W]
        video = (video * 255.).numpy().astype(np.uint8)
        return wandb.Video(video, fps=10, caption=caption)

@torch.no_grad()
def extract_video_slots(model, dataset):
    """Returns slots extracted from each video of the dataset."""
    model.eval()
    slot_key = 'post_slots' if params.model == 'StoSAVi' else 'slots'
    torch.cuda.empty_cache()
    # videos are long, so we use 1 video per GPU as 1 batch
    bs = torch.cuda.device_count()
    all_slots = []
    range_idx = range(0, dataset.num_videos, bs)

    #sampling the video
    visualizer = SAViMethod()
    visualizer._sample_video(model, dataset)

    for start_idx in tqdm(range_idx):
        end_idx = min(start_idx + bs, dataset.num_videos)
        data_dict = default_collate(
            [dataset.get_video(i) for i in range(start_idx, end_idx)])
        in_dict = {'img': data_dict['video'].float().cuda()}
        out_dict = model(in_dict)
        slots = out_dict[slot_key].detach().cpu().numpy()  # [B, T, n, c]
        all_slots += [slot for slot in slots]
        torch.cuda.empty_cache()
    all_slots = np.stack(all_slots, axis=0)  # [N, T, n, c]
    return all_slots


def process_video(model):
    """Extract slot_embs using video SlotAttn model"""
    train_set, val_set = build_dataset(params)

    # forward through train/val dataset
    print(f'Processing {params.dataset} video val set...')
    val_slots = extract_video_slots(model, val_set)
    print(f'Processing {params.dataset} video train set...')
    train_slots = extract_video_slots(model, train_set)

    # also extract test_set for CLEVRER
    if params.dataset == 'clevrer':
        test_set = build_clevrer_dataset(params, test_set=True)
        print(f'Processing {params.dataset} video test set...')
        test_slots = extract_video_slots(model, test_set)

    try:
        if('icswm' in params.dataset):
            train_slots = {
                i: train_slots[i] for i in range(len(train_slots))
            }
            val_slots = {
                i: val_slots[i] for i in range(len(val_slots))
            }
        else:
            train_slots = {
                os.path.basename(train_set.files[i]): train_slots[i]
                for i in range(len(train_slots))
            }  # each embedding is of shape [T, N, C]
            val_slots = {
                os.path.basename(val_set.files[i]): val_slots[i]
                for i in range(len(val_slots))
            }
        slots = {'train': train_slots, 'val': val_slots}

        if params.dataset == 'clevrer':
            test_slots = {
                os.path.basename(test_set.files[i]): test_slots[i]
                for i in range(len(test_slots))
            }
            slots['test'] = test_slots
        mkdir_or_exist(os.path.dirname(args.save_path))
        dump_obj(slots, args.save_path)
        print(args.save_path)
        print(f'Finish {params.dataset} video dataset, '
              f'train: {len(train_slots)}/{train_set.num_videos}, '
              f'val: {len(val_slots)}/{val_set.num_videos}')

        if params.dataset == 'clevrer':
            print(f'test: {len(test_slots)}/{test_set.num_videos}')
    except:
        pdb.set_trace()

    # create soft link to the weight dir
    if 'physion' in args.params:
        ln_path = os.path.join(
            os.path.dirname(args.weight), f'{args.subset}_slots.pkl')
    else:
        ln_path = os.path.join(
            os.path.dirname(args.weight), 'slots.pkl')
    os.system(r'ln -s {} {}'.format(args.save_path, ln_path))


def process_test_video(model):
    """Extract slot_embs using video SlotAttn model"""
    test_set = build_dataset(params)

    # forward through test dataset
    print(f'Processing {params.dataset} video test set...')
    test_slots = extract_video_slots(model, test_set)

    try:
        test_slots = {
            os.path.basename(test_set.files[i]): test_slots[i]
            for i in range(len(test_slots))
        }  # each embedding is of shape [T, N, C]
        slots = {'test': test_slots}
        mkdir_or_exist(os.path.dirname(args.save_path))
        dump_obj(slots, args.save_path)
        print(args.save_path)
        print(f'Finish {params.dataset} video dataset, '
              f'test: {len(test_slots)}/{test_set.num_videos}')
    except:
        pdb.set_trace()

    # create soft link to the weight dir
    ln_path = os.path.join(
            os.path.dirname(args.weight), 'test_slots.pkl')
    os.system(r'ln -s {} {}'.format(args.save_path, ln_path))


def main():
    model = build_model(params)
    model.load_state_dict(
        torch.load(args.weight, map_location='cpu')['state_dict'])
    model.testing = True  # we just want slots
    model = torch.nn.DataParallel(model).cuda().eval()
    if 'test' in params.dataset:
        pass
        #process_test_video(model)
    else:
        process_video(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract slots from videos')
    parser.add_argument('--params', type=str, required=True)
    parser.add_argument('--subset', type=str, default='training')  # Physion
    parser.add_argument(
        '--weight', type=str, required=True, help='pretrained model weight')
    parser.add_argument(
        '--save_path',
        type=str,
        required=True,  # './data/CLEVRER/slots.pkl'
        help='path to save slots',
    )

    args = parser.parse_args()

    if args.params.endswith('.py'):
        args.params = args.params[:-3]
    sys.path.append(os.path.dirname(args.params))
    params = importlib.import_module(os.path.basename(args.params))
    params = params.SlotFormerParams()
    if 'physion' in args.params:
        params.dataset = f'physion_{args.subset}'
    # assert params.dataset in args.save_path

    wandb_path = (os.sep).join(args.save_path.split(os.sep)[:-1])
    wandb.init(
        project='extract_slots',
        name='logger',
        id=None,
        dir=wandb_path
    )
    torch.backends.cudnn.benchmark = True
    main()
