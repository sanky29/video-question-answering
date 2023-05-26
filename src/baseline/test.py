from model import VQA
from dataloader import get_dataloader
from torchsummary import summary
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
        self.data.append(get_dataloader(args, mode = 'test'))
        self.results = []

        #the output dir
        if(self.args.output_file is None):
            self.args.output_file = os.path.join(self.args.resume_dir, 'test_results.txt')
        
        #get the model
        self.model = VQA(self.args).to(self.device)#, normalization)
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
    
        losses = []

        #with torch no grad
        with torch.no_grad():
            self.model.eval()

            #run for batch
            predicted_labels = []
            true_labels = []

            pbar = tqdm(self.data, ncols = 110, bar_format = "{l_bar}%s{bar:50}%s{r_bar}" % (Fore.BLUE, Fore.RESET))
            pbar.set_postfix({"loss":100})
            i = 0
            for data in pbar:
            
                predicted_label = self.model(data)
                loss = self.loss(data['labels'], predicted_label)

                #append loss to the file
                losses.append(loss.item())

                #append to the list
                prob = predicted_label
                predicted_label = predicted_label.view(-1, self.args.classifier.out_dim)
                predicted_label = torch.argmax(predicted_label, dim = 1).detach().cpu().numpy().tolist()
                true_label = data['labels'].view(-1).detach().cpu().numpy().tolist()
                
                predicted_labels.extend(predicted_label)
                true_labels.extend(true_label)

                #update the progress bar
                pbar.set_postfix({"loss":loss.item()})
                pbar.update(1)       
        
        results = get_metric(true_labels, predicted_labels, self.args.classifier.out_dim, indv = True)
        results['loss'] = sum(losses)/len(losses)
        
        self.print_result(results)
        return results

