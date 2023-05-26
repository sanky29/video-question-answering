from model import VQA
from dataloader_v2 import get_dataloader
# from torchsummary import summary
from tqdm import tqdm 
from torch.nn.functional import one_hot
from config import dump_to_file, load_config
from utils import get_metric

import pdb
import torch
import os
import sys

class Tester:

    def __init__(self,args):

        #store the args
        self.args = args

        #load the config
        self.args = load_config(os.path.join(self.args.resume_dir, 'config.yaml'), self.args)

        #get the meta data
        self.device = self.args.device
        
        #get the data sets
        self.data = get_dataloader(args, mode = 'train')
        self.data = [self.data[1],self.data[1]]
        self.data.append(get_dataloader(args, mode = 'test'))
        self.results = []

        #the output dir
        if(self.args.output_file is None):
            self.args.output_file = os.path.join(self.args.resume_dir, 'test_results.txt')
        
        #get the model
        self.model = ResNet(self.args.n, self.args.r).to(self.device)#, normalization)
        self.restore()


    #restore the original model
    def restore(self):
        checkpoint_name = self.args.checkpoint
        if(checkpoint_name is None):
            checkpoints = os.listdir(os.path.join(self.args.resume_dir, 'checkpoint'))
            checkpoints.sort()
            if('best.pt' in checkpoints):
                checkpoint_name = 'best.pt'
            else:
                checkpoint_name = checkpoints[-1]
        checkpoint_path = os.path.join(self.args.resume_dir, 'checkpoint', checkpoint_name)
        self.model.load_state_dict(torch.load(checkpoint_path))
        print(f'LOADED {checkpoint_path}')

    def print_result(self, result):
        for k in result.keys():
            print(k, result[k])
    
    def save_result(self, result):
        with open(self.args.output_file, "w") as f:
            for k in result.keys():
                f.write(f'{k}: {result[k]}')
            f.close()

    def test(self):
        
        #we need to return the average loss
        predicted_labels = []
        true_labels = []

        with torch.no_grad():

            #set in the train mode
            self.model.eval()

            #run for batch

            for data in self.data:
                for (data, true_label) in tqdm(data):
                    
                    #predict the labels 
                    predicted_label = self.model(data)
                    predicted_label = torch.argmax(predicted_label, dim = 1)
                    predicted_label = predicted_label.detach().cpu().tolist()
                    true_label = true_label.cpu().tolist()
                    
                    predicted_labels.extend(predicted_label)
                    true_labels.extend(true_label)

                result = get_metric(true_labels, predicted_labels, self.args.r)
                self.save_result(result)
                # self.print_result(result)
                self.results.append(result['f1_micro'])
                self.results.append(result['f1_macro'])
                self.results.append(result['accuracy'])
            res = ""
            for i in self.results:
                res += str(round(i, 3)) + "&"
            return res[:-1]
                
            