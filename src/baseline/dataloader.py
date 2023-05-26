import pickle
import pdb
import torch 
import numpy as np 
import os
import h5py
from caption_dataloader import CaptionLoader
from torch.utils.data import DataLoader, random_split


'''
get the video given the video id
'''
class VideoData:

    def __init__(self, files):

        #get the total files
        self.files = files
        self.files.sort()

        #read files
        self.batches = []
        for f in self.files:
            data_file = h5py.File(f, "r")
            self.batches.append(data_file)

        #total data:
        self.len = 0 
        for batch in self.batches:
            self.len += len(batch)

        #per batch data
        self.per_batch = self.len//len(self.batches)

    def __getitem__(self, index):
        return torch.tensor(np.array(self.batches[index // self.per_batch][str(index)][::8,:])).permute(0,3,1,2)/255

    def __len__(self):
        return self.len

'''
the overall dataset
'''
class Data:
    
    def __init__(self, video_files, question_file, tokenizer, questions_per_video, transform=None):

        #get the corresponding datasets
        self.video_data = VideoData(video_files)
        self.question_data = CaptionLoader(question_file, tokenizer, questions_per_video, transform)

        #all possible videos
        self.videos = list(self.question_data.video_question_mapping.keys())

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        '''
        video: [T x 3 x H x W]
        questions: list of tokenized questions
        mask: [QUESTIONS_PER_VIDEO x OUTPUT_DIM]
        labels: [QUESTIONS_PER_VIDEO]
        '''
        #get questions
        vid = self.videos[index]
        questions, attn_mask, mask, labels = self.question_data[vid]

        #get the video
        video = self.video_data[vid]
        
        return video, questions, attn_mask, mask, labels


def collate_fn(device, data):
    #TODO: add the batching of tokens here 
    #data: list of tuple (video, text, mask, label)
    ret = dict()
    ret['video'] = torch.cat([t[0].unsqueeze(0) for t in data], dim = 0).to(device)
    question = torch.cat([t[1].unsqueeze(0) for t in data], dim = 0).to(device)
    attn_mask = torch.cat([t[2].unsqueeze(0) for t in data], dim = 0).to(device)
    ret['ques'] = {'input_ids': question, 'attention_mask':attn_mask}
    ret['mask'] = torch.cat([t[3].unsqueeze(0) for t in data], dim = 0).to(device)
    ret['labels'] = torch.cat([t[4].unsqueeze(0) for t in data], dim = 0).to(device)
    return ret


def get_dataloader(args, tokenizer = None, mode = 'train'):

    #shuffle when only need to train
    #get arguments

    #1. video files
    video_files = os.listdir(args[mode].frame_dir)
    video_files = [os.path.join(args[mode].frame_dir, video_file) for video_file in video_files]

    #question_file
    question_file = args[mode].question_file

    #questions_per_video
    questions_per_video = args[mode].questions_per_video

    #get the data
    _collate_fn = lambda x: collate_fn(args.device, x)
    data = Data(video_files, question_file, tokenizer, questions_per_video)     
    
    val_data_len = int(len(data)*args.val_split)
    train_data, val_data = random_split(data, (len(data)-val_data_len, val_data_len))
    
    train_data = DataLoader(train_data, batch_size = args.train.batch_size, shuffle = True, collate_fn = _collate_fn, num_workers = args.workers)
    val_data = DataLoader(val_data, batch_size = args.val.batch_size, shuffle = False, collate_fn = _collate_fn, num_workers = args.workers)
    
    return train_data, val_data







'''
the overall dataset
'''
class TestData:
    
    def __init__(self, video_files, question_file, tokenizer, transform=None):

        #get the corresponding datasets
        self.video_data = VideoData(video_files)
        self.question_data = CaptionLoader(question_file, tokenizer, -1, transform, mode = 'test')

        #all possible videos
        self.videos = list(self.question_data.video_question_mapping.keys())

    def __len__(self):
        return len(question_data)

    def __getitem__(self, index):
        '''
        video: [T x 3 x H x W]
        questions: list of tokenized questions
        mask: [QUESTIONS_PER_VIDEO x OUTPUT_DIM]
        labels: [QUESTIONS_PER_VIDEO]
        '''
        #get questions
        vid, questions, attn_mask, mask, labels = self.question_data.get_test(idx)

        #get the video
        video = self.video_data[vid]
        
        return video, questions, attn_mask, mask, labels


def collate_fn_test(device, data):
    #TODO: add the batching of tokens here 
    #data: list of tuple (video, text, mask, label)
    ret = dict()
    ret['video'] = torch.cat([t[0].unsqueeze(0) for t in data], dim = 0).to(device)
    question = torch.cat([t[1].unsqueeze(0) for t in data], dim = 0).unsqueeze(1).to(device)
    attn_mask = torch.cat([t[2].unsqueeze(0) for t in data], dim = 0).unsqueeze(1).to(device)
    ret['ques'] = {'input_ids': question, 'attention_mask':attn_mask}
    ret['mask'] = torch.cat([t[3].unsqueeze(0) for t in data], dim = 0).unsqueeze(1).to(device)
    ret['labels'] = torch.cat([t[4].unsqueeze(0) for t in data], dim = 0).unsqueeze(1).to(device)
    return ret






def get_dataloader_test(args, tokenizer = None):

    #shuffle when only need to train
    #get arguments
    mode = 'test'

    #1. video files
    video_files = os.listdir(args[mode].frame_dir)
    video_files = [os.path.join(args[mode].frame_dir, video_file) for video_file in video_files]

    #question_file
    question_file = args[mode].question_file

    #get the data
    _collate_fn = lambda x: collate_fn_test(args.device, x)
    data = TestData(video_files, question_file, tokenizer)     
    
    test_data = DataLoader(val_data, batch_size = 1, shuffle = False, collate_fn = _collate_fn, num_workers = args.workers)
    
    return train_data, val_data