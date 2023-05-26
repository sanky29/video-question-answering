from train import Trainer
from test import Tester
from preprocess import VideoPreprocessor
from config import get_config
from argparse import ArgumentParser
import torch.multiprocessing as mp

def get_parser():
    
    #the parser for the arguments
    parser = ArgumentParser(
                        prog = 'python main.py',
                        description = 'This is main file for COL775 A1. You can train model from scratch or resume training, test checkpoint from this file',
                        epilog = 'thank you!!')

    #there are two tasks ['train', 'test']
    parser.add_argument('--task', choices=['train', 'test', 'preprocess_video'], default = 'train', required=False)

    #there are two tasks ['train', 'test']
    parser.add_argument('--args', nargs='+', required=False,  default = [], 
                        help='arguments for the config file in the form [key value] eg. dataset gravity epoch 200')

    return parser

if __name__ == '__main__':

    #parse the arguments
    parser = get_parser()
    args = parser.parse_args()

    mp.set_start_method('spawn')
    
    #get the task and updated args
    task = args.task
    args = get_config(args.args)

    if(task == 'train'):
        trainer = Trainer(args)
        trainer.train()
    
    if(task == 'test'):
        tester = Tester(args)
        tester.test()
    
    if(task == 'preprocess_video'):
        preprocessor = VideoPreprocessor(args, mode = 'test')
        preprocessor.preprocess()