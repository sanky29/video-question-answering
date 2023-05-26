from model import VQA
from dataloader import get_dataloader
from utils import get_metric
from torchsummary import summary
from tqdm import tqdm 
from torch.nn.functional import one_hot
from config import dump_to_file
from torch.optim import Adam
import pdb
import torch
import os
import sys
from colorama import Fore
import torch.nn as nn
class Trainer:

    def __init__(self,args):

        #store the args
        self.args = args
        self.clean_args()

        #get the meta data
        self.epochs = self.args.epochs
        self.epoch = 0
        self.steps = 0
        self.backprop_every = self.args.train.batch_size//4
        self.lr = self.args.lr
        self.device = self.args.device
        self.init_dirs()
        
        #validation and checkpoint frequency
        self.validate_every = self.args.validate_every
        self.checkpoint_every = self.args.checkpoint_every
        
        #get the model
        #self.model = #, normalization)
        model = VQA(self.args).to(self.device)
        #self.model= nn.DataParallel(model).to(self.device)
        self.model = model.to(self.device)

        #get the data
        tokenizer = model.tokenizer
        self.train_data, self.val_data = get_dataloader(self.args, tokenizer = tokenizer)
        
        #the minimum validation loss
        self.optimizer = self.get_optimizer()
    
    '''clean args'''
    def clean_args(self):
        if(self.args.video_models.arch == 'lstm'):
            self.args.video_models.out_dim = self.args.video_models.lstm.hidden_dim
        elif(self.args.video_models.arch == 'transformer'):
            self.args.video_models.out_dim = self.args.video_models.transformer.input_dim    
        else:
            self.args.video_models.out_dim = self.args.video_models.resnet3d.hidden_dim
        
    def init_dirs(self):

        #create the root dirs
        self.root = os.path.join(self.args.root, self.args.experiment_name)
        if(not os.path.exists(self.root)):
            os.makedirs(self.root)
        self.checkpoint_dir = os.path.join(self.root, 'checkpoint')
        if(not os.path.exists(self.checkpoint_dir)):
            os.makedirs(self.checkpoint_dir)
        self.train_loss_file = os.path.join(self.root, 'train_loss.txt')
        with open(self.train_loss_file, "w") as f:
            pass
        self.val_loss_file = os.path.join(self.root, 'val_loss.txt')
        with open(self.val_loss_file, "w") as f:
            pass
        self.args_file = os.path.join(self.root, 'config.yaml')
        dump_to_file(self.args_file, self.args)
        
    def save_model(self, best = False):
        if(best):
            checkpoint_name = f'best.pt'
        else:
            checkpoint_name = f'model_{self.epoch_no}.pt'
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        torch.save(self.model.state_dict(), checkpoint_path)
    
    def save_results(self, results, validate = False):
        if(validate):
            file = self.val_loss_file
        else:
            file = self.train_loss_file
        with open(file, "a") as f:
            s = ''
            for loss in results:
                s += str(results[loss]) + ','
            f.write(f'{s}\n')
            f.close()

    def get_optimizer(self):
        bert_param, resnet_param, rest = self.get_parameters()
        params = [{"params": bert_param, "lr": 0.001},
            # {'params': resnet_param, 'lr': 0.001},
            {'params': rest, 'lr': 0.001}]
        return Adam(params)
            
    def get_parameters(self):
        bert_param = []
        resnet_param = []
        rest = []
        for name, param in self.model.named_parameters():
            if('text_encoder' in name):
                bert_param.append(param)
            elif('frame_encoder' in name):
                resnet_param.append(param)
            else:
                rest.append(param)
        return bert_param, resnet_param, rest

    def print_results(self, results, validate = False):
        
        #print type of results
        print()
        if(validate):
            print("----VALIDITAION----")
        else:
            print(f'----EPOCH: {self.epoch_no}----')       
        for i in results:
            print(f'{i}: {results[i]} |')
        print()
        print("-"*20)

    def loss(self, true_label, predicted_labels):
        
        #add loss
        predicted_labels = predicted_labels + 1e-15
        true_label = one_hot(true_label, num_classes = self.args.classifier.out_dim)
        
        #loss: B x OSL x VOCAB_DICT  
        loss = -1*true_label*torch.log(predicted_labels)
        return loss.sum(-1).mean()

    #validate function is just same as the train 
    def validate(self):
        
        #we need to return the average loss
        losses = []

        #with torch no grad
        with torch.no_grad():
            self.model.eval()

            #run for batch
            predicted_labels = []
            true_labels = []

            pbar = tqdm(self.val_data, ncols = 110, bar_format = "{l_bar}%s{bar:50}%s{r_bar}" % (Fore.BLUE, Fore.RESET))
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

                if(i == 0):
                    print(predicted_label)
                    print(true_label)
                    print(prob[0])
                i += 1

                #update the progress bar
                pbar.set_postfix({"loss":loss.item()})
                pbar.update(1)       
        
        results = get_metric(true_labels, predicted_labels, self.args.classifier.out_dim)
        results['loss'] = sum(losses)/len(losses)
        return results


    def train_epoch(self):
        
        #we need to return the average loss
        losses = []
        
        #set in the train mode
        self.model.train()

        #run for batch
        #with tqdm(total=len(self.train_data)) as t:   
        pbar = tqdm(self.train_data, ncols = 110, bar_format = "{l_bar}%s{bar:50}%s{r_bar}" % (Fore.GREEN, Fore.RESET))
        pbar.set_postfix({"loss":100})
        for data in pbar:
            pdb.set_trace()
            self.optimizer.zero_grad()
            predicted_labels = self.model(data)
            loss = self.loss(data['labels'], predicted_labels)
            
            #if(self.steps % self.backprop_every == 0):
            #do backpropogation
            loss.backward()
            #pdb.set_trace()
            self.optimizer.step()
            
            #append loss to the file
            losses.append(loss.item())
            
            pbar.set_postfix({"loss":loss.item()})
            pbar.update(1)

            
            self.steps += 1
            #update the progress bar
            # t.set_postfix(loss=f"{loss.item():.2f}")
            # t.update(1) .
                   
        
        return {'loss': sum(losses)/len(losses)}

    def train(self):

        torch.autograd.set_detect_anomaly(True)
        
        for epoch in tqdm(range(self.epochs), ncols = 110, bar_format = "{l_bar}%s{bar:50}%s{r_bar}" % (Fore.YELLOW, Fore.RESET)):
            self.epoch_no = epoch

            #train for one epoch and print results
            train_loss = self.train_epoch()
            self.print_results(train_loss)
            self.save_results(train_loss)            

            #checkpoint the model
            if(self.epoch_no % self.checkpoint_every == 0):
                self.save_model()

            #do validation if neccesary
            if(self.epoch_no % self.validate_every == 0):
                val_loss = self.validate()
                self.print_results(val_loss, validate = True)
                self.save_results(val_loss, validate = True)

