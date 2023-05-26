import cv2 as cv
import numpy as np
from tqdm import tqdm
import os
import h5py
import pdb
import multiprocessing

class VideoPreprocessor:

    '''
    the video directory looks like
    data_dir: video0-1000/video_000.mp4
    '''
    def __init__(self, args, mode = 'train'):
        self.args = args
        
        #final and data dirs
        self.video_dir = self.args[mode].video_dir
        self.frame_dir = self.args[mode].frame_dir
        if(not os.path.exists(self.frame_dir)):
            os.makedirs(self.frame_dir)

        #total batches
        self.batches = 0
    
    def preprocess_video(self,path):
        '''
        Args:
            path: <path_to_video>/<video_name>.mp4
        Returns:
            frames: [T x 3 x H x W]
        '''
        #the video buffer
        buffer = cv.VideoCapture(path)
        
        #dimesnions
        frame_count = int(buffer.get(cv.CAP_PROP_FRAME_COUNT))
        frame_width = int(buffer.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(buffer.get(cv.CAP_PROP_FRAME_HEIGHT))

        #frames: [frame_count x frame_height x frame_width] 
        frames = np.zeros((frame_count//5, frame_height, frame_width, 3))

        #read frames
        for i in range(0, frame_count, 5):
            _, frame = buffer.read()
            frames[i] = frame.astype('float')
            for t in range(5):
                _, frame = buffer.read()    

        return frames
    
    '''
    preprocess a whole batch
    '''
    def preprocess_batch(self, batch, batch_id):

        #create
        videos = os.listdir(batch)
        
        #create file
        data_file = os.path.join(self.frame_dir, f'{batch_id}.hdf5')
        data_file = h5py.File(data_file, "w")

        t = set()
        #pre process video
        for video in tqdm(videos):
            
            #get the frames
            frames = self.preprocess_video(os.path.join(batch,video))

            #get the video id
            video_id = int(video.split('.')[0][6:])
            data_file[str(video_id)] = frames
        
        print(f"{batch_id} completed")
        self.batches += 1
        
    def preprocess(self):

        #process one batch at a time
        batches = os.listdir(self.video_dir)
        batches.sort()
        print(batches)
        from multiprocessing import set_start_method
        set_start_method('fork')
        
        #run batch process in parallel
        for batch_id, batch in tqdm(enumerate(batches)):
            
            #batch dirs
            batch_dir = os.path.join(self.video_dir, batch)
            
            #the process list
            process_list = []

            #create process
            process = multiprocessing.Process(target = self.preprocess_batch, 
                args = (batch_dir, batch_id))
            
            #start and join
            process.start()
            process_list.append(process)
            
        #wait for end
        for process in process_list:
            process.join()

        
'''
for every
    head_1(x): 8
    head_2(x): 2
[10]
mask
batching

1. rescale
2. same video
3. video wise [same update]
    Effective_b = B*PER_VIDEO_Q
    for video[B x LENGTH_OF_VIDEOx 3 x H x W], q[B x PER_VIDEO_Q x LEN] in data:
        video_enc = video_model(video) [B x VIDEO_ENC].expand(PER_VIDEO_Q)
        output = head(video_enc, question_model(q))
    back_prop
    
    problem addressed: extrem B = 1 [SAME UPDATE]
                       extrem PER_VIDEO_Q  = 1 [VERY SMALL BATCH]
                       inbetween: 


n: batch size bound
'''